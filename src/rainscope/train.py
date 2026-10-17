"""Run once with frozen config; never tune against the final test report."""

import argparse
import json
import platform
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .data import load_data, xy
from .models import make_model
from .evaluation import metrics, choose_threshold, ResourceMonitor, bootstrap_by_month


def run(root, config_path=None):
    root = Path(root)
    cfg = json.loads(Path(config_path or root / "config.json").read_text())
    report = root / "reports"
    report.mkdir(exist_ok=True)
    private = report / "private"
    private.mkdir(exist_ok=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    parts, audit = load_data(root / "data/raw/weatherAUS.csv", cfg)
    (report / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit["splits"], indent=2), flush=True)
    x, y = xy(parts["train"])
    xv, yv = xy(parts["validation"])
    xt, yt = xy(parts["test"])
    models = {}
    rows = []
    failures = {}
    resources = {}
    for name in cfg["models"]:
        print("Fitting " + name, flush=True)
        try:
            with threadpool_limits(limits=cfg["workers"]):
                with ResourceMonitor() as monitor:
                    if name == "tabm":
                        from .modern import TabMModel

                        model = TabMModel(cfg).fit(x, y, xv, yv)
                    elif name == "tabicl_v2":
                        from .modern import TabICLModel

                        model = TabICLModel(cfg).fit(x, y, xv, yv)
                    else:
                        model = make_model(name, cfg).fit(x, y)
                fit_resources = monitor.result
                with ResourceMonitor() as monitor:
                    pv = model.predict_proba(xv)[:, 1]
                resources[name] = {
                    "fit": fit_resources,
                    "validation_predict": monitor.result,
                }
            threshold = choose_threshold(yv, pv, cfg["threshold_grid"])
            score = metrics(yv, pv, threshold)
            rows.append(
                {
                    "model": name,
                    "threshold": threshold,
                    "training_rows": getattr(model, "training_rows", len(y)),
                    **{k: v for k, v in score.items() if k != "confusion_matrix"},
                }
            )
            models[name] = model
            print(
                name + " validation AP=" + str(score["average_precision"]), flush=True
            )
            if hasattr(model, "history"):
                (report / "tabm_learning_curve.json").write_text(
                    json.dumps(model.history, indent=2)
                )
        except Exception as exc:
            import traceback

            failures[name] = repr(exc)
            traceback.print_exc()
    if len(models) < 2:
        raise RuntimeError("Not enough successful models to compare")
    table = pd.DataFrame(rows).sort_values(
        ["average_precision", "model"], ascending=[False, True]
    )
    table.to_csv(report / "validation_leaderboard.csv", index=False)
    selected = str(table.iloc[0]["model"])
    threshold = float(table.iloc[0]["threshold"])
    # Persist selection BEFORE examining final test predictions.
    selection = {
        "selected_model": selected,
        "threshold": threshold,
        "criterion": "highest validation average precision; F1 threshold chosen on validation only",
        "config": cfg,
        "failures": failures,
        "data_sha256": audit["sha256"],
        "python": sys.version,
        "platform": platform.platform(),
    }
    (report / "selection.json").write_text(json.dumps(selection, indent=2))
    tests = []
    predictions = {}
    for name, model in models.items():
        print("Final test " + name, flush=True)
        with threadpool_limits(limits=cfg["workers"]):
            with ResourceMonitor() as monitor:
                pt = model.predict_proba(xt)[:, 1]
        resources[name]["test_predict"] = monitor.result
        own_threshold = float(table.loc[table.model == name, "threshold"].iloc[0])
        score = metrics(yt, pt, own_threshold)
        tests.append(
            {
                "model": name,
                "selected_by_validation": name == selected,
                "threshold": own_threshold,
                **{k: v for k, v in score.items() if k != "confusion_matrix"},
            }
        )
        predictions[name] = pt
    pd.DataFrame(tests).to_csv(report / "test_leaderboard.csv", index=False)
    pt = predictions[selected]
    model = models[selected]
    score = metrics(yt, pt, threshold)
    score["month_cluster_intervals"] = bootstrap_by_month(
        parts["test"], yt, pt, threshold, cfg["bootstrap_repeats"], cfg["seed"]
    )
    (report / "test_metrics.json").write_text(json.dumps(score, indent=2))
    (report / "resources.json").write_text(json.dumps(resources, indent=2))
    pd.DataFrame(
        {
            "Date": parts["test"].Date,
            "Location": parts["test"].Location,
            "actual": yt,
            "probability": pt,
            "predicted": pt >= threshold,
        }
    ).to_csv(private / "test_predictions.csv", index=False)
    slices = []
    for location, sub in (
        parts["test"].assign(actual=yt, probability=pt).groupby("Location")
    ):
        if len(sub) >= 100 and sub.actual.nunique() == 2:
            m = metrics(sub.actual.to_numpy(), sub.probability.to_numpy(), threshold)
            slices.append(
                {
                    "station": location,
                    "rows": len(sub),
                    "rain_rate": sub.actual.mean(),
                    **{
                        k: m[k]
                        for k in [
                            "average_precision",
                            "f1",
                            "recall",
                            "precision",
                            "brier",
                        ]
                    },
                }
            )
    pd.DataFrame(slices).to_csv(report / "station_metrics.csv", index=False)
    # Save fitted full pipelines locally; training observations/checkpoints are not published.
    joblib.dump(
        {"model": model, "selection": selection}, artifacts / "selected_model.joblib"
    )
    joblib.dump(
        {"models": models, "selection": selection}, artifacts / "all_models.joblib"
    )
    if selected != "tabicl_v2":
        print("Validation permutation importance", flush=True)
        # Validation-only: explainability must not drive changes using test labels.
        rng = np.random.default_rng(cfg["seed"])
        ix = rng.choice(len(xv), min(2000, len(xv)), replace=False)
        base = average_precision_score(yv[ix], model.predict_proba(xv.iloc[ix])[:, 1])
        imps = []
        for col in xv.columns:
            z = xv.iloc[ix].copy()
            z[col] = rng.permutation(z[col].to_numpy())
            drop = base - average_precision_score(yv[ix], model.predict_proba(z)[:, 1])
            imps.append({"feature": col, "validation_ap_drop": drop})
        pd.DataFrame(imps).sort_values("validation_ap_drop", ascending=False).to_csv(
            report / "permutation_importance.csv", index=False
        )
    from .report import build_report

    build_report(root, parts, predictions, selected, threshold)
    print("Selected by validation:", selected, "test:", json.dumps(score), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--config")
    args = p.parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()
