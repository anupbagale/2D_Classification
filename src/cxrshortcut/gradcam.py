"""Grad-CAM heatmaps and how much of each heatmap falls inside the lung box.

If a model really reads the lungs, most of its heatmap mass should sit inside
the lung region. Uniform attention would put (box area) of the mass inside, so
that area is the "no preference" reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from .images import DEFAULT_LUNG_BOX, load_image, lung_box_mask, occlude_lungs
from .models import last_conv_layer_name


def gradcam(model: keras.Model, images, layer_name: str | None = None, class_indices=None) -> np.ndarray:
    """Grad-CAM for a batch. Returns heatmaps (N, H, W) scaled to [0, 1] per image.

    ``class_indices``: which class to explain per image; default is the predicted class.
    """
    layer_name = layer_name or last_conv_layer_name(model)
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.outputs[0]])
    images = tf.convert_to_tensor(images, tf.float32)
    with tf.GradientTape() as tape:
        conv, preds = grad_model(images, training=False)
        if class_indices is None:
            class_indices = tf.argmax(preds, axis=1)
        idx = tf.cast(tf.convert_to_tensor(class_indices), tf.int32)
        score = tf.gather(preds, idx, axis=1, batch_dims=1)
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))
    cam = tf.image.resize(cam[..., None], images.shape[1:3])[..., 0]
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy()


def lung_fraction(cams: np.ndarray, box=DEFAULT_LUNG_BOX) -> np.ndarray:
    """Share of each heatmap's total mass inside the lung box (NaN for an all-zero map)."""
    cams = np.asarray(cams, dtype=np.float64)
    mask = lung_box_mask(cams.shape[1], cams.shape[2], box)
    total = cams.sum(axis=(1, 2))
    inside = (cams * mask).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, inside / total, np.nan)


def box_area_fraction(box=DEFAULT_LUNG_BOX, size: int = 224) -> float:
    return float(lung_box_mask(size, size, box).mean())


def gradcam_report(
    model: keras.Model,
    df: pd.DataFrame,
    classes: list[str],
    out_dir: str | Path,
    img_size: int = 224,
    n_per_class: int = 40,
    occlude: bool = False,
    box=DEFAULT_LUNG_BOX,
    seed: int = 42,
    batch_size: int = 16,
    n_figure_per_class: int = 4,
) -> dict:
    """Grad-CAM on a stratified sample of ``df``; saves gradcam_examples.png."""
    sample = pd.concat(
        [g.sample(min(len(g), n_per_class), random_state=seed) for _, g in df.groupby("label")],
        ignore_index=True,
    )
    images = np.stack([load_image(p, img_size).numpy() for p in sample.path])
    if occlude:
        images = occlude_lungs(images, box)

    cams, probs = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        cams.append(gradcam(model, batch))
        probs.append(model.predict(batch, verbose=0))
    cams, probs = np.concatenate(cams), np.concatenate(probs)

    sample["predicted"] = [classes[i] for i in probs.argmax(1)]
    sample["lung_fraction"] = lung_fraction(cams, box)
    sample.drop(columns=["split"], errors="ignore").to_csv(Path(out_dir) / "gradcam_lung_fraction.csv", index=False)

    per_class = {
        label: {
            "n": int(len(g)),
            "n_nonzero_heatmaps": int(g.lung_fraction.notna().sum()),
            "mean_lung_fraction": float(g.lung_fraction.mean()),
            "median_lung_fraction": float(g.lung_fraction.median()),
        }
        for label, g in sample.groupby("label")
    }
    _save_figure(images, cams, sample, classes, box, Path(out_dir) / "gradcam_examples.png", n_figure_per_class)
    return {
        "box": list(box),
        "box_area_fraction": box_area_fraction(box, img_size),
        "mean_lung_fraction": float(sample.lung_fraction.mean()),
        "per_class": per_class,
    }


def _save_figure(images, cams, sample, classes, box, path, n_per_class) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = [sample.index[sample.label == c][:n_per_class].tolist() for c in classes]
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(len(classes), n_cols, figsize=(2.4 * n_cols, 2.6 * len(classes)), squeeze=False)
    size = images.shape[1]
    top, bottom, left, right = box
    for r, idxs in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(idxs):
                continue
            i = idxs[c]
            ax.imshow(images[i, ..., 0], cmap="gray", vmin=0, vmax=255)
            ax.imshow(cams[i], cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.add_patch(Rectangle((left * size, top * size), (right - left) * size, (bottom - top) * size,
                                   fill=False, edgecolor="white", linestyle="--", linewidth=0.8))
            s = sample.loc[i]
            ax.set_title(f"true {s.label}\npred {s.predicted} · in box {s.lung_fraction:.0%}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
