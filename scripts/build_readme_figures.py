"""Build presentation figures from training data and recorded experiment outputs.

Does not train models, select thresholds or alter the recorded evaluation.
"""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rainscope.data import load_data, NUMERIC

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
FIGURES = REPORT / "figures"
NAMES = {
    "climatology": "Climatology",
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram boosting",
    "tabm": "TabM",
    "tabicl_v2": "TabICLv2 (3k context)",
}


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    parts, _ = load_data(ROOT / "data/raw/weatherAUS.csv", cfg)
    train = parts["train"][NUMERIC].copy()
    train["RainTomorrow"] = (parts["train"].RainTomorrow == "Yes").astype(int)
    correlation = train.corr()  # Pairwise observed values; training partition only.
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(correlation)), correlation.index)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            value = correlation.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.6 else "#172b3a",
            )
    ax.set_title(
        "Weather feature correlations · training observations only",
        pad=18,
        fontweight="bold",
    )
    fig.colorbar(
        im, ax=ax, shrink=0.8, label="Pearson correlation (pairwise non-missing values)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = json.loads((REPORT / "test_metrics.json").read_text())
    selected = json.loads((REPORT / "selection.json").read_text())["selected_model"]
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.imshow(cm, cmap="Blues")
    labels = [
        ["True negatives", "False positives"],
        ["False negatives", "True positives"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}\n{labels[i][j]}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "#172b3a",
            )
    ax.set_xticks([0, 1], ["No rain", "Rain"])
    ax.set_yticks([0, 1], ["No rain", "Rain"])
    ax.set(
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{NAMES[selected]} · final holdout confusion matrix",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    table = pd.read_csv(REPORT / "test_leaderboard.csv").sort_values(
        "average_precision"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#197c87" if name != selected else "#e58332" for name in table.model]
    bars = ax.barh(
        [NAMES[name] for name in table.model], table.average_precision, color=colors
    )
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in table.average_precision], padding=5)
    ax.set(
        xlim=(0, 0.85),
        xlabel="Average precision · rain class",
        title="Model comparison · final chronological holdout",
    )
    ax.axvline(metrics["average_precision"], color="#e58332", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(FIGURES / "average_precision_comparison.png", dpi=150)
    plt.close(fig)
    # Reuse recorded scores and probabilities; no new model fitting or selection.
    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
        average_precision_score,
        f1_score,
    )
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, (key, title) in zip(
        axes.flat,
        [
            ("average_precision", "Average precision ↑"),
            ("roc_auc", "ROC-AUC ↑"),
            ("f1", "Rain F1 ↑"),
            ("precision", "Rain precision ↑"),
            ("recall", "Rain recall ↑"),
            ("brier", "Brier loss ↓"),
        ],
    ):
        bars = ax.barh([NAMES[n] for n in table.model], table[key], color=colors)
        ax.bar_label(
            bars, labels=[f"{v:.3f}" for v in table[key]], padding=4, fontsize=9
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 1.14 if key != "brier" else 0.22)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Six models · final holdout metric comparison", fontsize=16, fontweight="bold"
    )
    fig.savefig(FIGURES / "metric_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pred = pd.read_csv(REPORT / "private/test_predictions.csv", parse_dates=["Date"])
    y = pred.actual.to_numpy()
    prob = pred.probability.to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    precision, recall, _ = precision_recall_curve(y, prob)
    axes[0].plot(
        recall,
        precision,
        color="#e58332",
        lw=2,
        label=f"TabM · AP {metrics['average_precision']:.3f}",
    )
    axes[0].axhline(
        y.mean(), color="grey", ls="--", label=f"Rain prevalence {y.mean():.3f}"
    )
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    fpr, tpr, _ = roc_curve(y, prob)
    axes[1].plot(
        fpr, tpr, color="#197c87", lw=2, label=f"TabM · AUC {metrics['roc_auc']:.3f}"
    )
    axes[1].plot([0, 1], [0, 1], color="grey", ls="--", label="No-skill diagonal")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    observed, mean = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    axes[2].plot(mean, observed, "o-", color="#197c87", label="TabM · quantile bins")
    axes[2].plot([0, 1], [0, 1], color="grey", ls="--", label="Ideal reliability")
    axes[2].set(
        xlabel="Predicted probability",
        ylabel="Observed rain fraction",
        title="Probability reliability",
    )
    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.12)
    fig.suptitle("TabM · recorded final holdout predictions", fontweight="bold")
    fig.savefig(FIGURES / "tabm_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    history = pd.read_json(REPORT / "tabm_learning_curve.json")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        history.epoch,
        history.train_loss,
        label="Training loss (batch mean)",
        color="#197c87",
    )
    ax.plot(
        history.epoch,
        history.validation_log_loss,
        label="Validation log loss",
        color="#e58332",
    )
    # Fit retains a new checkpoint when validation loss improves by > 0.0001.
    best = float("inf")
    best_epoch = None
    for row in history.itertuples():
        if row.validation_log_loss < best - 0.0001:
            best = row.validation_log_loss
            best_epoch = row.epoch
    ax.axvline(
        best_epoch,
        ls="--",
        color="grey",
        label=f"Retained checkpoint: epoch {best_epoch}",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Binary cross-entropy",
        title="TabM learning curve · training and validation only",
    )
    ax.legend()
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=150)
    plt.close(fig)

    threshold = json.loads((REPORT / "selection.json").read_text())["threshold"]
    monthly = []
    for month, group in pred.groupby(pred.Date.dt.to_period("M")):
        monthly.append(
            {
                "month": str(month),
                "rows": len(group),
                "average_precision": average_precision_score(
                    group.actual, group.probability
                ),
                "f1": f1_score(
                    group.actual, group.probability >= threshold, zero_division=0
                ),
            }
        )
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(REPORT / "monthly_metrics.csv", index=False)
    stations = pd.read_csv(REPORT / "station_metrics.csv").sort_values(
        "average_precision"
    )
    extremes = pd.concat([stations.head(5), stations.tail(5)]).sort_values(
        "average_precision"
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    axes[0].plot(
        monthly.month,
        monthly.average_precision,
        "o-",
        label="Average precision",
        color="#197c87",
    )
    axes[0].plot(
        monthly.month,
        monthly.f1,
        "o-",
        label=f"F1 · fixed threshold {threshold:.2f}",
        color="#e58332",
    )
    axes[0].set_xticks(
        range(0, len(monthly), 2), monthly.month.iloc[::2], rotation=45, ha="right"
    )
    axes[0].set(ylim=(0, 1), ylabel="Score", title="TabM · performance by test month")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.12)
    bars = axes[1].barh(
        [f"{r.station} (n={r.rows})" for r in extremes.itertuples()],
        extremes.average_precision,
        color="#197c87",
    )
    axes[1].bar_label(
        bars,
        labels=[f"{v:.3f}" for v in extremes.average_precision],
        padding=4,
        fontsize=9,
    )
    axes[1].set(
        xlim=(0, 1.1),
        xlabel="Average precision",
        title="Station slices · five lowest and highest AP",
    )
    fig.suptitle(
        "Holdout error analysis · descriptive only, no retuning", fontweight="bold"
    )
    fig.savefig(FIGURES / "error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        "Generated heatmap, six-metric dashboard, diagnostic curves, confusion matrix, learning curve and error-analysis plots."
    )


if __name__ == "__main__":
    main()
