import math

import numpy as np
import pytest

from cxrshortcut.metrics import add_confidence_intervals, bootstrap_ci, compute_metrics

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA"]


def one_hot(idx, k=3, p=0.8):
    probs = np.full((len(idx), k), (1 - p) / (k - 1))
    probs[np.arange(len(idx)), idx] = p
    return probs


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 2, 2])
    m = compute_metrics(y, one_hot(y), CLASSES)
    assert m["accuracy"] == m["balanced_accuracy"] == m["macro_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_always_majority_class_exposes_imbalance():
    # 1 COVID, 2 NORMAL, 7 PNEUMONIA; model always says PNEUMONIA
    y = np.array([0, 1, 1] + [2] * 7)
    m = compute_metrics(y, one_hot(np.full(10, 2)), CLASSES)
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["majority_baseline_accuracy"] == pytest.approx(0.7)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["per_class"]["COVID19"]["sensitivity"] == 0
    assert m["per_class"]["PNEUMONIA"]["specificity"] == 0
    assert math.isnan(m["per_class"]["COVID19"]["ppv"])  # never predicted


def test_known_confusion_matrix():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    m = compute_metrics(y, one_hot(pred, k=2), ["COVID19", "NORMAL"])
    c = m["per_class"]["COVID19"]
    assert m["confusion_matrix"] == [[3, 1], [2, 2]]
    assert c["sensitivity"] == pytest.approx(0.75)
    assert c["specificity"] == pytest.approx(0.5)
    assert c["ppv"] == pytest.approx(0.6)
    assert m["balanced_accuracy"] == pytest.approx(0.625)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], np.zeros((2, 2)), CLASSES)


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    noisy = np.where(rng.random(200) < 0.8, y, rng.integers(0, 3, 200))
    probs = one_hot(noisy)
    point = compute_metrics(y, probs, CLASSES)["accuracy"]
    lo, hi = bootstrap_ci(y, probs, lambda a, p: compute_metrics(a, p, CLASSES)["accuracy"], n_boot=300)
    assert lo <= point <= hi and hi - lo < 0.2


def test_confidence_intervals_for_positive_class():
    y = np.array([0, 1, 2] * 20)
    m = add_confidence_intervals(compute_metrics(y, one_hot(y), CLASSES), y, one_hot(y), "COVID19", n_boot=50)
    assert m["ci95"]["COVID19_sensitivity"] == [1.0, 1.0]
    assert set(m["ci95"]) >= {"macro_auc", "balanced_accuracy", "COVID19_specificity"}
