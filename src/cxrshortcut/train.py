"""Training with a held-out validation set, class weights and early stopping.

Model selection (best epoch) uses validation AUC only; the test set is not seen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import keras

from .images import DEFAULT_LUNG_BOX, make_dataset
from .models import DENSENET_FINE_TUNE_FROM, build_model, set_backbone_trainable


def class_weights(labels: pd.Series, classes: list[str]) -> dict[int, float]:
    """Balanced weights n / (k * count) so the rare class (COVID) counts as much."""
    counts = labels.value_counts().reindex(classes, fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError("Every class needs at least one training image")
    n, k = counts.sum(), len(classes)
    return {i: float(n / (k * c)) for i, c in enumerate(counts)}


def _compile(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def train_model(
    model_name: str,
    df: pd.DataFrame,
    classes: list[str],
    out_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 20,
    fine_tune_epochs: int = 0,
    lr: float = 1e-3,
    fine_tune_lr: float = 1e-5,
    patience: int = 5,
    occlude: bool = False,
    box=DEFAULT_LUNG_BOX,
    seed: int = 42,
    weights: str | None = "imagenet",
    verbose: int = 2,
) -> tuple[keras.Model, pd.DataFrame]:
    """Train on split=='train', select the best epoch on split=='val'.

    Returns the best model (reloaded from its checkpoint) and the per-epoch history.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(seed)

    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    if train_df.empty or val_df.empty:
        raise ValueError("Need both train and val rows; call add_validation_split first")

    common = dict(classes=classes, size=img_size, batch_size=batch_size, occlude=occlude, box=box, seed=seed)
    train_ds = make_dataset(train_df, training=True, **common)
    val_ds = make_dataset(val_df, training=False, **common)

    model = build_model(model_name, img_size, len(classes), weights=weights)
    _compile(model, lr)

    ckpt = out_dir / "model.keras"
    log = out_dir / "history.csv"
    if log.exists():
        log.unlink()

    def callbacks(threshold=None):
        return [
            keras.callbacks.ModelCheckpoint(
                ckpt, monitor="val_auc", mode="max", save_best_only=True,
                initial_value_threshold=threshold,
            ),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience),
            keras.callbacks.CSVLogger(log, append=True),
        ]

    cw = class_weights(train_df.label, classes)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=cw,
                   callbacks=callbacks(), verbose=verbose, shuffle=False)  # tf.data already shuffles
    best = max(h1.history["val_auc"])

    if fine_tune_epochs > 0:
        if model_name != "densenet121":
            raise ValueError("Fine-tuning is only defined for densenet121")
        # Continue from the best head-only weights, then unfreeze the last dense block.
        model = keras.models.load_model(ckpt)
        set_backbone_trainable(model, unfreeze_from=DENSENET_FINE_TUNE_FROM)
        _compile(model, fine_tune_lr)
        start = len(h1.epoch)
        model.fit(train_ds, validation_data=val_ds, initial_epoch=start,
                  epochs=start + fine_tune_epochs, class_weight=cw,
                  callbacks=callbacks(threshold=best), verbose=verbose, shuffle=False)

    history = pd.read_csv(log)
    best_model = keras.models.load_model(ckpt)
    return best_model, history


def history_summary(history: pd.DataFrame) -> dict:
    best = int(np.nanargmax(history["val_auc"].to_numpy()))
    row = history.iloc[best]
    return {
        "epochs_run": int(len(history)),
        "best_epoch": int(row["epoch"]) + 1,
        "best_val_auc": float(row["val_auc"]),
        "best_val_accuracy": float(row["val_accuracy"]),
        "train_accuracy_at_best": float(row["accuracy"]),
    }
