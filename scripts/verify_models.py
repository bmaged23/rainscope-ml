"""Re-score trusted saved models with frozen data/splits/thresholds; never refit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from rainscope.data import load_data, xy
from rainscope.evaluation import (
    ResourceMonitor,
    bootstrap_by_month,
    choose_threshold,
    metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "reports/model_verification"
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "reports/selection.json").read_text())
    cfg = selection[
        "config"
    ]  # frozen experiment configuration, not edited serving config
    parts, audit = load_data(root / "data/raw/weatherAUS.csv", cfg)
    if audit["sha256"] != selection["data_sha256"]:
        raise RuntimeError("Dataset differs from the frozen experiment")
    artifact = root / "artifacts/all_models.joblib"
    bundle = joblib.load(artifact)
    if bundle["selection"] != selection:
        raise RuntimeError("Artifact and recorded model selection differ")
    validation = pd.read_csv(root / "reports/validation_leaderboard.csv").set_index(
        "model"
    )
    prior = pd.read_csv(root / "reports/test_leaderboard.csv").set_index("model")
    xv, yv = xy(parts["validation"])
    xt, yt = xy(parts["test"])
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "frozen fitted models, no retraining/tuning; same chronological holdout, not an independent new dataset",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "dataset_sha256": audit["sha256"],
        "splits": audit["splits"],
        "selected_model": selection["selected_model"],
        "models": {},
        "majority_no_rain_accuracy": float((yt == 0).mean()),
        "resource_limitations": "Models share one process and are all deserialized before timing. RSS and CUDA allocated memory include resident model/interpreter overhead, not isolated per-model requirements. CPU uses one core = 100%.",
        "limitations": "Single-seed replay on the already evaluated holdout; not new evidence of generalization. Month-cluster intervals are descriptive, not evidence of statistically significant superiority.",
    }
    rows = []
    for name in cfg["models"]:
        print("Verifying", name, flush=True)
        model = bundle["models"][name]
        with threadpool_limits(limits=cfg["workers"]):
            with ResourceMonitor() as monitor:
                pv = model.predict_proba(xv)[:, 1]
            validation_resources = monitor.result
            threshold = float(validation.loc[name, "threshold"])
            recomputed = choose_threshold(yv, pv, cfg["threshold_grid"])
            if not np.isclose(threshold, recomputed):
                raise RuntimeError(f"{name}: frozen threshold no longer reproduced")
            process = psutil.Process()
            cpu_before = sum(process.cpu_times()[:2])
            started = time.perf_counter()
            with ResourceMonitor() as monitor:
                pt = model.predict_proba(xt)[:, 1]
            elapsed = time.perf_counter() - started
            resource = monitor.result
            resource["cpu_percent_one_core_basis"] = (
                (sum(process.cpu_times()[:2]) - cpu_before) / elapsed * 100
            )
        for probabilities, expected_rows in [(pv, len(yv)), (pt, len(yt))]:
            if (
                len(probabilities) != expected_rows
                or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise RuntimeError(f"{name}: invalid probabilities")
        score = metrics(yt, pt, threshold)
        differences = {
            k: float(score[k] - prior.loc[name, k])
            for k in score
            if k != "confusion_matrix"
        }
        reproduced = all(abs(d) <= 1e-5 for d in differences.values())
        result = {
            "threshold": threshold,
            "threshold_reproduced": True,
            "metrics": score,
            "differences_from_recorded_metrics": differences,
            "recorded_metrics_reproduced_at_1e_5": reproduced,
            "validation_resources": validation_resources,
            "test_resources": resource,
            "uncertainty": bootstrap_by_month(
                parts["test"], yt, pt, threshold, cfg["bootstrap_repeats"], cfg["seed"]
            ),
        }
        report["models"][name] = result
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "reproduced": reproduced,
                **{k: v for k, v in score.items() if k != "confusion_matrix"},
                "test_predict_seconds": resource["wall_seconds"],
            }
        )
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        pd.DataFrame(rows).to_csv(output / "leaderboard.csv", index=False)
        print(
            name,
            "AP=",
            score["average_precision"],
            "reproduced=",
            reproduced,
            flush=True,
        )
    report["all_six_reproduced"] = len(report["models"]) == 6 and all(
        x["recorded_metrics_reproduced_at_1e_5"] for x in report["models"].values()
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_six_reproduced"]:
        raise RuntimeError("At least one model did not reproduce its recorded metrics")
    print("All six model results reproduced. Saved", output, flush=True)


if __name__ == "__main__":
    main()
