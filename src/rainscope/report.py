from pathlib import Path
import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve


def build_report(root, parts, predictions, selected, threshold):
    report = Path(root) / "reports"
    figures = report / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    y = (parts["test"].RainTomorrow == "Yes").astype(int).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, p in predictions.items():
        precision, recall, _ = precision_recall_curve(y, p)
        axes[0].plot(recall, precision, label=name)
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, label=name)
        actual, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[2].plot(mean, actual, "o-", label=name)
    axes[0].axhline(y.mean(), ls="--", color="grey")
    axes[0].set(
        xlabel="Recall", ylabel="Precision", title="Rain-class precision–recall"
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(
        xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curves"
    )
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].set(
        xlabel="Mean predicted probability",
        ylabel="Observed rain fraction",
        title="Reliability (quantile bins)",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
    )
    fig.suptitle("RainScope ML · Final chronological holdout", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    audit = json.loads((report / "data_audit.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    missing = pd.Series(audit["missing_percent"]).sort_values().tail(10)
    missing.plot.barh(ax=axes[0], color="#187b89")
    axes[0].set(title="Raw-data missingness", xlabel="Missing (%)")
    data = pd.concat(list(parts.values()))
    year = (
        data.assign(
            Year=data.Date.dt.year, Rain=(data.RainTomorrow == "Yes").astype(int)
        )
        .groupby("Year")
        .Rain.mean()
    )
    year.plot(ax=axes[1], marker="o", color="#e98735")
    axes[1].axvline(2015, color="grey", ls="--")
    axes[1].axvline(2016, color="grey", ls="--")
    axes[1].set(title="Rain prevalence by observation year", ylabel="Rain fraction")
    fig.tight_layout()
    fig.savefig(figures / "data_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    importance = report / "permutation_importance.csv"
    if importance.exists():
        imp = pd.read_csv(importance).head(12).sort_values("validation_ap_drop")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(imp.feature, imp.validation_ap_drop, color="#187b89")
        ax.set(
            title="Validation permutation importance",
            xlabel="Average precision decrease (one permutation; 2,000 validation rows)",
        )
        fig.tight_layout()
        fig.savefig(figures / "feature_importance.png", dpi=150)
        plt.close(fig)
    val = pd.read_csv(report / "validation_leaderboard.csv")
    test = pd.read_csv(report / "test_leaderboard.csv")
    selection = json.loads((report / "selection.json").read_text())
    m = json.loads((report / "test_metrics.json").read_text())

    def table(df, cols):
        out = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            out.append(
                "| "
                + " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
                + " |"
            )
        return "\n".join(out)

    text = (
        f"""# Recorded experiment results

Selected on validation average precision: **{selected}**. Decision threshold: **{threshold:.2f}**, chosen for validation F1. Test results below were not used to select the model or threshold. This is a bounded single-seed comparison, not an exhaustive model ranking.

## Data partitions

"""
        + table(
            pd.DataFrame([dict(split=k, **v) for k, v in audit["splits"].items()]),
            ["split", "rows", "start", "end", "rain_rate"],
        )
        + """

Two-day embargoes separate observations across split boundaries. All learned preprocessing is fitted on training only. Same weather stations can appear across time; this does not measure generalization to unseen stations.

## Validation comparison

"""
        + table(
            val,
            [
                "model",
                "training_rows",
                "average_precision",
                "roc_auc",
                "f1",
                "threshold",
            ],
        )
        + """

## Final test comparison (diagnostic, not selection)

"""
        + table(
            test,
            [
                "model",
                "selected_by_validation",
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1",
                "brier",
            ],
        )
        + f"""

Average precision is used directly, not trapezoidal PR-AUC. Positive-class metrics refer to rain. TabICLv2 receives a training-only stratified context capped at {selection["config"]["tabicl_context_rows"]} rows to fit a 6 GB GPU; other models use the complete training partition. Thus this is a resource-constrained comparison, not an equal-data benchmark. TabM early stopping monitors validation log loss.

![Diagnostic curves](figures/model_comparison.png)

![Data quality](figures/data_quality.png)

## Uncertainty and resources

Selected model test AP: {m["average_precision"]:.4f}; F1: {m["f1"]:.4f}. Month-cluster bootstrap intervals: AP {m["month_cluster_intervals"]["average_precision"]}; F1 {m["month_cluster_intervals"]["f1"]}. These descriptive intervals do not establish robustness to unseen locations or future climate regimes.

[Resource measurements](resources.json) report fit/predict wall time, peak process RSS, mean process CPU percentage (may exceed 100% across cores) and peak PyTorch CUDA allocation. Process RSS includes the interpreter baseline and resident preceding models. These are not whole-system measurements. GPU allocation also includes other resident models: nonzero CUDA memory during a CPU estimator phase does not mean that estimator requires a GPU. Sub-100ms CPU measurements are coarse. Foundation-model prediction performs most in-context computation. [Station slices](station_metrics.csv) expose geographic variation; do not optimize on test slices.

## Processing analysis

- Drop missing target labels; never impute the target.
- Preserve numeric missingness with train-fitted median imputation and missing indicators.
- Fit categorical encoders on training only; handle unknown categories at inference.
- Derive cyclical month and within-day pressure/humidity changes without future observations.
- Keep plausible extremes; avoid arbitrary full-dataset clipping or balancing the test set.
- Exclude raw Date from model inputs; split before fitting transforms.
- Reject conflicting station/date duplicates and malformed numeric/date input.

This experiment uses one engineered feature set. It does not isolate the causal benefit of individual features. Permutation importance is validation-only and descriptive. Daily measurements can accumulate through midnight: this is historical daily-record classification, not a validated morning weather service. Foundation-model exposure to this public dataset during pretraining cannot be ruled out. No improvement over the upstream random-split notebook is claimed because protocols differ.

## Failed candidates

```json
{json.dumps(selection["failures"], indent=2)}
```
"""
    )
    (report / "RESULTS.md").write_text(text)
