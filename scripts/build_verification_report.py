"""Render the completed frozen-model replay into a readable table and plot."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
folder = root / "reports/model_verification"
report = json.loads((folder / "verification.json").read_text())
assert report["all_six_reproduced"]
audit = json.loads((folder / "preprocessing_audit.json").read_text())
assert len(audit) == 6 and all(r["verified"] for r in audit.values())
items = sorted(
    report["models"].items(), key=lambda x: x[1]["metrics"]["average_precision"]
)
labels = {
    "climatology": "Climatology",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram boosting",
    "tabm": "TabM",
    "tabicl_v2": "TabICLv2 (3k context)",
}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
for ax, metric, title in zip(
    axes, ["average_precision", "f1"], ["Average precision", "Rain-class F1"]
):
    for i, (name, r) in enumerate(items):
        score = r["metrics"][metric]
        low, high = r["uncertainty"][metric]
        ax.hlines(
            i, low, high, color="#e58e26" if name == "tabm" else "#457b9d", linewidth=3
        )
        ax.scatter(
            score, i, color="#e58e26" if name == "tabm" else "#457b9d", s=40, zorder=3
        )
    ax.set_yticks(np.arange(len(items)), [labels[n] for n, _ in items])
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
fig.suptitle(
    "Frozen-model replay: 25,974 chronological holdout records\nPoints: scores; bars: descriptive 95% month-cluster bootstrap intervals",
    fontsize=12,
)
fig.savefig(folder / "quality_verification.png", dpi=160)
plt.close(fig)
lines = [
    "# Frozen-model verification — 22 October 2026",
    "",
    "**All six saved models reproduced their recorded holdout metrics within absolute tolerance 1e-5.** No models were retrained or tuned. Dataset SHA-256 and saved selection matched the original experiment; validation-selected thresholds reproduced. This is a reproducibility check on an already evaluated holdout, not a new independent benchmark.",
    "",
    "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | Brier ↓ | Test prediction time |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name, r in sorted(
    report["models"].items(), key=lambda x: -x[1]["metrics"]["average_precision"]
):
    m = r["metrics"]
    lines.append(
        f"| {labels[name]} | {m['accuracy']:.2%} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['average_precision']:.3f} | {m['brier']:.3f} | {r['test_resources']['wall_seconds']:.3f} s |"
    )
lines += [
    "",
    "![Quality replay and descriptive uncertainty](quality_verification.png)",
    "",
    "## Protocol and interpretation",
    "",
    "- Training: 98,892 observations through 2014; validation: 17,136 observations in 2015; test: 25,974 observations from 2016–2017. Two-day split embargo retained. Station identities overlap across periods.",
    "- Numeric imputation medians, scaler sample counts and categorical vocabularies match training-only data for all models. TabICLv2 matches its deterministic 3,000-row stratified training context.",
    "- TabM remains the validation-selected model and has the highest AP and lowest Brier loss in this replay. The intervals do not establish statistically significant superiority.",
    f"- An always-no-rain classifier would reach {report['majority_no_rain_accuracy']:.2%} accuracy with zero rain recall. The climatology row instead uses its frozen F1-selected threshold, predicting rain throughout; do not compare accuracy alone.",
    "- TabICLv2 trades a small training context for expensive prediction. Its short historical fit time is not its request latency.",
    "- Resource readings are process-level, with all saved models loaded before scoring. RSS and allocated CUDA memory include resident models and interpreter overhead; they are not isolated model memory requirements. CPU percentages use one core as 100%. HTTP benchmarks are a separate workload.",
    "- Bootstrap bars resample whole test months 200 times. They describe temporal variability for this particular historical split; this remains a single-seed experiment.",
    "",
    "## Reproduce",
    "",
    "Use the existing project environment with modern CUDA dependencies and trusted local artifacts:",
    "",
    "```bash",
    "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/verify_models.py",
    "python scripts/verify_preprocessing.py",
    "python scripts/build_verification_report.py",
    "```",
    "",
    "[Full metrics, confusion matrices, resource readings and intervals](verification.json) · [Training-preprocessing audit](preprocessing_audit.json) · [CSV leaderboard](leaderboard.csv)",
    "",
    "The source dataset and fitted artifacts remain local. These scripts verify trusted models created by this project; joblib files from untrusted sources must not be loaded.",
]
(folder / "README.md").write_text("\n".join(lines) + "\n")
print("Rendered", folder / "README.md")
