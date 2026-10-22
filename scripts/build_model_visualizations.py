"""Plot recorded training history and verified test results; no refitting or tuning."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
OUT = REPORT / "figures"
NAMES = {
    "tabm": "TabM",
    "hist_gradient_boosting": "Histogram boosting",
    "random_forest": "Random forest",
    "logistic": "Logistic regression",
    "tabicl_v2": "TabICLv2 (3k context)",
    "climatology": "Climatology",
}
BLUE, ORANGE = "#367ca5", "#e58e26"


def main():
    OUT.mkdir(exist_ok=True)
    verified = json.loads((REPORT / "model_verification/verification.json").read_text())
    if not verified.get("all_six_reproduced"):
        raise RuntimeError("Complete model verification before rendering these figures")
    validation = pd.read_csv(REPORT / "validation_leaderboard.csv").set_index("model")
    resources = json.loads((REPORT / "resources.json").read_text())
    history = pd.DataFrame(
        json.loads((REPORT / "tabm_learning_curve.json").read_text())
    )
    order = list(NAMES)
    colors = [ORANGE if n == "tabm" else BLUE for n in order]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    ax = axes[0, 0]
    ax.plot(
        history.epoch,
        history.train_loss,
        label="Training batch loss (dropout on)",
        color=BLUE,
    )
    ax.plot(
        history.epoch,
        history.validation_log_loss,
        label="Validation log loss",
        color=ORANGE,
    )
    # Replay the actual minimum-improvement rule used by TabMModel.fit.
    best_loss, retained_epoch = float("inf"), None
    for row in history.itertuples():
        if row.validation_log_loss < best_loss - 0.0001:
            best_loss, retained_epoch = row.validation_log_loss, row.epoch
    ax.axvline(
        retained_epoch,
        ls="--",
        color="#555",
        label=f"Retained epoch {int(retained_epoch)}",
    )
    ax.set(
        title="TabM optimization history — recorded 40 epochs",
        xlabel="Epoch",
        ylabel="Loss (lower is better)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    ax = axes[0, 1]
    counts = [int(validation.loc[n, "training_rows"]) for n in order]
    ax.barh([NAMES[n] for n in order], counts, color=colors)
    ax.invert_yaxis()
    for i, v in enumerate(counts):
        ax.text(v + 1500, i, f"{v:,}", va="center", fontsize=9)
    ax.set(
        title="Training observations / foundation-model context",
        xlabel="Rows",
        xlim=(0, max(counts) * 1.25),
    )
    ax = axes[1, 0]
    fit = [resources[n]["fit"]["wall_seconds"] for n in order]
    ax.barh([NAMES[n] for n in order], fit, color=colors)
    ax.invert_yaxis()
    for i, v in enumerate(fit):
        ax.text(v + 0.3, i, f"{v:.2f} s", va="center", fontsize=9)
    ax.set(
        title="Original experiment: measured fit duration",
        xlabel="Wall seconds",
        xlim=(0, max(fit) * 1.25),
    )
    ax = axes[1, 1]
    values = [validation.loc[n, "average_precision"] for n in order]
    ax.barh([NAMES[n] for n in order], values, color=colors)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set(
        title="Model selection: 2015 validation AP only",
        xlabel="Average precision (higher is better)",
        xlim=(0, 1),
    )
    fig.suptitle(
        "Training and validation — frozen experiment protocol",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(OUT / "training_process.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    y = np.arange(len(order))
    for ax, key, title in zip(
        axes.flat,
        ["average_precision", "f1", "roc_auc", "brier"],
        ["Average precision ↑", "Rain-class F1 ↑", "ROC-AUC ↑", "Brier loss ↓"],
    ):
        v = [validation.loc[n, key] for n in order]
        t = [verified["models"][n]["metrics"][key] for n in order]
        ax.barh(y - 0.18, v, height=0.34, label="Validation (2015)", color=BLUE)
        ax.barh(y + 0.18, t, height=0.34, label="Test (2016–2017)", color=ORANGE)
        ax.set_yticks(y, [NAMES[n] for n in order])
        ax.invert_yaxis()
        for i, (a, b) in enumerate(zip(v, t)):
            ax.text(a + 0.008, i - 0.18, f"{a:.3f}", va="center", fontsize=8)
            ax.text(b + 0.008, i + 0.18, f"{b:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 0.225 if key == "brier" else 1.05)
        ax.grid(axis="x", alpha=0.15)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Validation versus verified chronological test performance\nDifferent periods and rain prevalence; no tuning on test data",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUT / "validation_vs_test.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, name in zip(axes.flat, order):
        matrix = np.asarray(verified["models"][name]["metrics"]["confusion_matrix"])
        rates = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(rates, vmin=0, vmax=1, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}\n({rates[i, j]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if rates[i, j] > 0.55 else "#172b3a",
                )
        threshold = verified["models"][name]["threshold"]
        ax.set(
            title=f"{NAMES[name]} · threshold {threshold:.2f}",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No rain", "Rain"],
            yticklabels=["No rain", "Rain"],
            xlabel="Predicted",
            ylabel="Actual",
        )
    fig.suptitle(
        "Test confusion matrices — counts and within-actual-class percentages\n25,974 observations; validation-selected thresholds held fixed",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUT / "all_models_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    offsets = {
        "tabm": (12, 8),
        "hist_gradient_boosting": (12, 10),
        "random_forest": (12, -17),
        "logistic": (12, -18),
        "tabicl_v2": (-135, 12),
        "climatology": (12, 10),
    }
    for name in order:
        r = verified["models"][name]
        seconds = r["test_resources"]["wall_seconds"]
        ap = r["metrics"]["average_precision"]
        ax.scatter(
            seconds,
            ap,
            s=110 if name == "tabm" else 70,
            color=ORANGE if name == "tabm" else BLUE,
            zorder=3,
        )
        ax.annotate(
            f"{NAMES[name]}\n{seconds:.3f} s · AP {ap:.3f}",
            (seconds, ap),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.03, 300)
    ax.set_ylim(0.18, 0.82)
    ax.set(
        xlabel="Predict all 25,974 test rows: wall seconds (log scale; lower is faster)",
        ylabel="Test average precision (higher is better)",
        title="Quality versus prediction time — verified saved-model replay",
    )
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "quality_latency_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Rendered four training/test figures in", OUT)


if __name__ == "__main__":
    main()
