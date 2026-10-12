"""Classification metrics that do not hide class imbalance, with bootstrap CIs.

Pure numpy/scikit-learn, so the dataset audit can use it without TensorFlow.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def _auc(y_binary: np.ndarray, score: np.ndarray) -> float:
    if y_binary.min() == y_binary.max():
        return float("nan")
    return float(roc_auc_score(y_binary, score))


def compute_metrics(y_true, probs, classes: list[str]) -> dict:
    """Accuracy plus the numbers that matter with imbalanced classes.

    ``y_true``: integer class indices, shape (N,).
    ``probs``: predicted probabilities, shape (N, K), columns in ``classes`` order.
    Predictions are the argmax class.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    k = len(classes)
    if probs.shape != (len(y_true), k):
        raise ValueError(f"probs shape {probs.shape} != ({len(y_true)}, {k})")

    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(k)))
    n = int(cm.sum())

    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        sens = _div(tp, tp + fn)
        ppv = _div(tp, tp + fp)
        per_class[name] = {
            "support": int(tp + fn),
            "sensitivity": sens,
            "specificity": _div(tn, tn + fp),
            "ppv": ppv,
            "f1": _div(2 * ppv * sens, ppv + sens) if not math.isnan(ppv + sens) else float("nan"),
            "auc_one_vs_rest": _auc((y_true == i).astype(int), probs[:, i]),
        }

    sensitivities = [v["sensitivity"] for v in per_class.values()]
    aucs = [v["auc_one_vs_rest"] for v in per_class.values()]
    counts = np.bincount(y_true, minlength=k)
    return {
        "n": n,
        "classes": list(classes),
        "accuracy": _div(np.trace(cm), n),
        "balanced_accuracy": float(np.nanmean(sensitivities)),
        "majority_baseline_accuracy": _div(counts.max(), n),
        "macro_auc": float(np.nanmean(aucs)) if not all(map(math.isnan, aucs)) else float("nan"),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(y_true, probs, statistic, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for ``statistic(y_true, probs) -> float``.

    Resamples that make the statistic undefined (e.g. a class missing) are skipped.
    Returns (low, high), or (nan, nan) if too few resamples were valid.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = statistic(y_true[idx], probs[idx])
        if v is not None and not math.isnan(v):
            values.append(v)
    if len(values) < max(10, n_boot // 2):
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def add_confidence_intervals(
    metrics: dict, y_true, probs, positive_class: str | None, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Add 95% bootstrap CIs for macro AUC, balanced accuracy and the positive
    class's sensitivity/specificity (e.g. COVID19)."""
    classes = metrics["classes"]

    def macro_auc(y, p):
        return compute_metrics(y, p, classes)["macro_auc"]

    def balanced_acc(y, p):
        return compute_metrics(y, p, classes)["balanced_accuracy"]

    ci = {
        "macro_auc": bootstrap_ci(y_true, probs, macro_auc, n_boot, seed=seed),
        "balanced_accuracy": bootstrap_ci(y_true, probs, balanced_acc, n_boot, seed=seed),
    }
    if positive_class is not None and positive_class in classes:
        i = classes.index(positive_class)

        def sens(y, p):
            pos = y == i
            return _div((p[pos].argmax(1) == i).sum(), pos.sum())

        def spec(y, p):
            neg = y != i
            return _div((p[neg].argmax(1) != i).sum(), neg.sum())

        ci[f"{positive_class}_sensitivity"] = bootstrap_ci(y_true, probs, sens, n_boot, seed=seed)
        ci[f"{positive_class}_specificity"] = bootstrap_ci(y_true, probs, spec, n_boot, seed=seed)

    out = dict(metrics)
    out["ci95"] = {k: list(v) for k, v in ci.items()}
    out["positive_class"] = positive_class
    return out
