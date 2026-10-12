"""Tiny fake chest-X-ray dataset for tests and for trying the pipeline without
downloading anything. The images are cartoons, not X-rays.

Each class has a pattern inside the lung region. With ``shortcut=True`` the
COVID19 class also gets a source "fingerprint" like a real mixed dataset:
different file format, colour mode, size and a bright corner marker.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ("COVID19", "NORMAL", "PNEUMONIA")


def _draw(label: str, rng: np.random.Generator, size: int) -> Image.Image:
    a = rng.normal(40, 8, (size, size)).clip(0, 255)
    img = Image.fromarray(a.astype(np.uint8), "L")
    d = ImageDraw.Draw(img)
    s = size
    # Two lungs
    d.ellipse([0.15 * s, 0.15 * s, 0.47 * s, 0.82 * s], fill=int(rng.integers(90, 110)))
    d.ellipse([0.53 * s, 0.15 * s, 0.85 * s, 0.82 * s], fill=int(rng.integers(90, 110)))
    if label == "COVID19":  # scattered bright spots in both lungs
        for _ in range(8):
            x, y = rng.uniform(0.2, 0.8) * s, rng.uniform(0.25, 0.75) * s
            r = 0.04 * s
            d.ellipse([x - r, y - r, x + r, y + r], fill=200)
    elif label == "PNEUMONIA":  # one dense patch in one lung
        x0 = 0.18 if rng.random() < 0.5 else 0.56
        d.rectangle([x0 * s, 0.45 * s, (x0 + 0.25) * s, 0.75 * s], fill=190)
    return img


def make_synthetic_dataset(
    root: str | Path,
    n_train: int = 20,
    n_test: int = 8,
    size: int = 96,
    shortcut: bool = True,
    duplicate: bool = True,
    seed: int = 0,
    classes=CLASSES,
) -> Path:
    """Write root/Data/{train,test}/<class>/*.png|jpg and return root/Data.

    ``duplicate=True`` copies the first training image of the first class into
    its test folder, so the duplicate check has something to find.
    """
    rng = np.random.default_rng(seed)
    data = Path(root) / "Data"
    if data.exists():
        shutil.rmtree(data)
    for split, n in (("train", n_train), ("test", n_test)):
        for label in classes:
            folder = data / split / label
            folder.mkdir(parents=True)
            for i in range(n):
                img = _draw(label, rng, size)
                if shortcut and label == "COVID19":
                    img = img.resize((int(size * 1.3), size)).convert("RGB")
                    ImageDraw.Draw(img).rectangle([2, 2, 12, 12], fill=(255, 255, 255))
                    img.save(folder / f"{label}_{i:03d}.png")
                else:
                    img.save(folder / f"{label}_{i:03d}.jpg", quality=90)
    if duplicate:
        src = sorted((data / "train" / classes[0]).iterdir())[0]
        shutil.copy(src, data / "test" / classes[0] / f"dup_{src.name}")
    return data
