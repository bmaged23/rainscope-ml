import time
import threading
import numpy as np
import psutil
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)


def metrics(y, p, threshold=0.5):
    pred = np.asarray(p) >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(
            log_loss(y, np.column_stack([1 - np.asarray(p), p]), labels=[0, 1])
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def choose_threshold(y, p, grid):
    # Ties go to the threshold closest to .5, then the smaller threshold.
    return float(
        max(
            grid,
            key=lambda t: (
                f1_score(y, np.asarray(p) >= t, zero_division=0),
                -abs(t - 0.5),
                -t,
            ),
        )
    )


class ResourceMonitor:
    """Process RSS/CPU samples and PyTorch peak allocated CUDA memory."""

    def __enter__(self):
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.samples = []
        self.stop = threading.Event()
        self.process.cpu_percent(None)
        try:
            import torch

            self.torch = torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self.torch = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            self.samples.append(
                (self.process.memory_info().rss, self.process.cpu_percent(None))
            )
            self.stop.wait(0.1)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.result = {
            "wall_seconds": time.perf_counter() - self.started,
            "peak_process_rss_mb": max(
                [s[0] for s in self.samples] or [self.process.memory_info().rss]
            )
            / 2**20,
            "mean_process_cpu_percent": float(np.mean([s[1] for s in self.samples]))
            if self.samples
            else 0.0,
            "cuda_peak_allocated_mb": self.torch.cuda.max_memory_allocated() / 2**20
            if self.torch and self.torch.cuda.is_available()
            else 0.0,
        }


def bootstrap_by_month(frame, y, p, threshold, repeats=200, seed=42):
    """Cluster resampling preserves within-month dependence, not a universal CI."""
    months = frame.Date.dt.to_period("M").astype(str).to_numpy()
    unique = np.unique(months)
    groups = [np.flatnonzero(months == m) for m in unique]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(repeats):
        ix = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if len(np.unique(y[ix])) < 2:
            continue
        out.append(
            [
                average_precision_score(y[ix], p[ix]),
                f1_score(y[ix], p[ix] >= threshold, zero_division=0),
            ]
        )
    low, high = np.quantile(out, [0.025, 0.975], axis=0)
    return {
        "method": "month-cluster percentile bootstrap; descriptive temporal uncertainty",
        "repeats": len(out),
        "average_precision": [float(low[0]), float(high[0])],
        "f1": [float(low[1]), float(high[1])],
    }
