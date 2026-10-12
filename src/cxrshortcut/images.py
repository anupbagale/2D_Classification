"""Image loading, augmentation and lung-region occlusion as tf.data pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# Approximate lung region as fractions of image height/width: (top, bottom, left, right).
# A fixed box is a crude stand-in for real lung masks; see the README's limitations.
DEFAULT_LUNG_BOX = (0.10, 0.85, 0.12, 0.88)

AUTOTUNE = tf.data.AUTOTUNE


def load_gray(path, size: int) -> tf.Tensor:
    """Read any supported image, convert to grayscale and resize: uint8 (size, size, 1).

    Grayscale first means RGB vs grayscale files (a possible source difference
    between classes) look the same to the model. uint8 keeps the in-memory cache small.
    """
    raw = tf.io.read_file(path)
    # Decode in the file's own channel layout (TensorFlow refuses channels=1 for
    # BMP and GIF), then reduce to one gray channel: drop alpha, convert colour.
    img = tf.io.decode_image(raw, channels=0, expand_animations=False)
    img = tf.cond(
        tf.shape(img)[-1] >= 3,
        lambda: tf.image.rgb_to_grayscale(img[..., :3]),
        lambda: img[..., :1],
    )
    img = tf.image.resize(img, (size, size), antialias=True)
    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    img.set_shape((size, size, 1))
    return img


def gray_to_model_input(img) -> tf.Tensor:
    """uint8 (..., H, W, 1) -> float32 (..., H, W, 3) in [0, 255]."""
    return tf.image.grayscale_to_rgb(tf.cast(img, tf.float32))


def load_image(path, size: int) -> tf.Tensor:
    """Model-ready image: float32 (size, size, 3) in [0, 255]."""
    return gray_to_model_input(load_gray(path, size))


_JPEG_PNG_MAGIC = (b"\xff\xd8\xff", b"\x89PNG")


def find_unreadable(df: pd.DataFrame) -> list[str]:
    """Paths TensorFlow cannot decode (e.g. 1-bit BMPs or corrupt files).

    JPEG and PNG files (recognised by their first bytes, not their extension)
    are trusted; anything else is test-decoded. This takes seconds, and stops a
    single odd file from crashing a run after training.
    """
    bad = []
    for path in df["path"]:
        with open(path, "rb") as f:
            head = f.read(4)
        if head.startswith(_JPEG_PNG_MAGIC):
            continue
        try:
            load_image(str(path), 8)
        except (tf.errors.OpError, ValueError):
            bad.append(str(path))
    return bad


def lung_box_mask(height: int, width: int, box=DEFAULT_LUNG_BOX) -> np.ndarray:
    """1 inside the lung box, 0 outside, shape (height, width)."""
    top, bottom, left, right = box
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"Invalid box {box}")
    mask = np.zeros((height, width), dtype=np.float32)
    r0, r1 = int(round(top * height)), int(round(bottom * height))
    c0, c1 = int(round(left * width)), int(round(right * width))
    mask[r0:r1, c0:c1] = 1.0
    return mask


def occlude_lungs(images, box=DEFAULT_LUNG_BOX):
    """Black out the lung box. Works on (H, W, C) or (N, H, W, C) arrays/tensors."""
    h, w = images.shape[-3], images.shape[-2]
    keep = (1.0 - lung_box_mask(h, w, box))[..., None]
    return images * keep


def build_augmenter(seed: int | None = None) -> keras.Sequential:
    """Mild, anatomically plausible augmentation.

    No flips: a chest X-ray is never upside down, and a left-right flip moves
    the heart to the wrong side.
    """
    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.02, fill_mode="constant", seed=seed),
            keras.layers.RandomZoom(0.1, fill_mode="constant", seed=seed),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="cxr_augment",
    )


def make_dataset(
    df: pd.DataFrame,
    classes: list[str],
    size: int = 224,
    batch_size: int = 32,
    training: bool = False,
    occlude: bool = False,
    box=DEFAULT_LUNG_BOX,
    seed: int = 42,
    cache: bool = True,
) -> tf.data.Dataset:
    """Batches of (images, one-hot labels).

    With ``cache=True`` the decoded, resized grayscale images are kept in memory
    after the first pass (about 50 KB per image at 224 px), so later epochs skip
    reading and resizing the full-size files. With ``training=False`` the order
    matches ``df`` exactly, so predictions can be joined back to the rows.
    """
    index = {c: i for i, c in enumerate(classes)}
    labels = df["label"].map(index)
    if labels.isna().any():
        unknown = sorted(set(df.loc[labels.isna(), "label"]))
        raise ValueError(f"Labels not in classes: {unknown}")

    paths = df["path"].astype(str).tolist()
    n_classes = len(classes)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype("int32").to_numpy()))
    ds = ds.map(lambda p, y: (load_gray(p, size), y), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: (gray_to_model_input(x), tf.one_hot(y, n_classes)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    if training:
        augment = build_augmenter(seed)
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 255.0), y),
            num_parallel_calls=AUTOTUNE,
        )
    if occlude:
        keep = tf.constant((1.0 - lung_box_mask(size, size, box))[None, ..., None])
        ds = ds.map(lambda x, y: (x * keep, y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
