"""Index an image-folder dataset (train/ and test/ with one sub-folder per class)
and make a stratified validation split from the training images.

The test folder is never used for training or model selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
TEST_DIR_NAMES = ("test", "testing")


def find_split_dirs(root: str | Path) -> tuple[Path, Path]:
    """Find the train and test folders anywhere under ``root`` (case-insensitive).

    Works for layouts like ``root/train``, ``root/Data/train`` or the Kaggle
    download folder, as long as train and test are siblings.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    trains = sorted(
        (p for p in candidates if p.name.lower() == "train"),
        key=lambda p: (len(p.parts), str(p)),
    )
    for train_dir in trains:
        for sibling in sorted(train_dir.parent.iterdir()):
            if sibling.is_dir() and sibling.name.lower() in TEST_DIR_NAMES:
                return train_dir, sibling
    raise FileNotFoundError(f"No sibling train/ and test/ folders found under {root}")


def index_split(split_dir: str | Path, split: str) -> pd.DataFrame:
    """One row per image: path, label (the class folder name) and split."""
    split_dir = Path(split_dir)
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(f), "label": class_dir.name, "split": split})
    return pd.DataFrame(rows, columns=["path", "label", "split"])


def index_dataset(root: str | Path, classes: list[str] | None = None) -> pd.DataFrame:
    """Index train and test images. Optionally keep only ``classes``."""
    train_dir, test_dir = find_split_dirs(root)
    df = pd.concat(
        [index_split(train_dir, "train"), index_split(test_dir, "test")],
        ignore_index=True,
    )
    if df.empty:
        raise ValueError(f"No images found under {root}")

    train_classes = set(df.loc[df.split == "train", "label"])
    test_classes = set(df.loc[df.split == "test", "label"])
    if train_classes != test_classes:
        raise ValueError(
            f"Train and test classes differ: train={sorted(train_classes)}, "
            f"test={sorted(test_classes)}"
        )
    if classes:
        missing = set(classes) - train_classes
        if missing:
            raise ValueError(f"Classes not found in data: {sorted(missing)}")
        df = df[df.label.isin(classes)]
    return df.reset_index(drop=True)


def add_validation_split(
    df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Move a stratified ``val_fraction`` of the training rows to split='val'."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    train = df[df.split == "train"]
    _, val_idx = train_test_split(
        train.index,
        test_size=val_fraction,
        stratify=train.label,
        random_state=seed,
    )
    out = df.copy()
    out.loc[val_idx, "split"] = "val"
    return out


def class_names(df: pd.DataFrame) -> list[str]:
    """Class names in a fixed (sorted) order; index i is output unit i."""
    return sorted(df.label.unique())


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """Images per class (rows) and split (columns)."""
    order = [s for s in ("train", "val", "test") if s in set(df.split)]
    return (
        df.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(columns=order)
    )
