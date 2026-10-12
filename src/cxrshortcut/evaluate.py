"""Test-set evaluation: predictions joined to rows, metrics with 95% CIs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import keras

from .images import DEFAULT_LUNG_BOX, make_dataset
from .metrics import add_confidence_intervals, compute_metrics


def predict_probs(
    model: keras.Model,
    df: pd.DataFrame,
    classes: list[str],
    img_size: int = 224,
    batch_size: int = 32,
    occlude: bool = False,
    box=DEFAULT_LUNG_BOX,
) -> np.ndarray:
    ds = make_dataset(df, classes, size=img_size, batch_size=batch_size, training=False, occlude=occlude, box=box)
    return model.predict(ds.map(lambda x, y: x), verbose=0)


def evaluate_model(
    model: keras.Model,
    df: pd.DataFrame,
    classes: list[str],
    out_dir: str | Path,
    tag: str = "test",
    img_size: int = 224,
    batch_size: int = 32,
    occlude: bool = False,
    box=DEFAULT_LUNG_BOX,
    positive_class: str | None = "COVID19",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Predict on ``df``, save ``predictions_<tag>.csv`` and return metrics with CIs."""
    probs = predict_probs(model, df, classes, img_size, batch_size, occlude, box)
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    preds = df[["path", "label"]].reset_index(drop=True).copy()
    for i, c in enumerate(classes):
        preds[f"p_{c}"] = probs[:, i]
    preds["predicted"] = [classes[i] for i in probs.argmax(1)]
    preds.to_csv(Path(out_dir) / f"predictions_{tag}.csv", index=False)

    metrics = compute_metrics(y, probs, classes)
    if positive_class not in classes:
        positive_class = None
    return add_confidence_intervals(metrics, y, probs, positive_class, n_boot=n_boot, seed=seed)
