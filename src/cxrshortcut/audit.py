"""Dataset audit that needs no deep learning (runs on CPU in a few minutes).

1. Metadata shortcut test: can a classifier that only sees file metadata
   (size, format, colour mode, overall brightness) predict the diagnosis?
   If it can, the classes come from different sources and a CNN can learn
   the source instead of the disease.
2. Duplicate check: identical or near-identical images shared between the
   training and test sets inflate test scores.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_metrics

NUMERIC_FEATURES = [
    "width",
    "height",
    "aspect",
    "file_kb",
    "bytes_per_pixel",
    "mean",
    "std",
    "frac_black",
    "frac_white",
]
CATEGORICAL_FEATURES = ["format", "mode"]

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- metadata


def image_metadata(path: str | Path) -> dict:
    p = Path(path)
    size_bytes = p.stat().st_size
    with Image.open(p) as im:
        fmt = im.format or p.suffix.lstrip(".").upper()
        mode = im.mode
        w, h = im.size
        gray = im.convert("L")
        gray.thumbnail((256, 256))
        a = np.asarray(gray, dtype=np.float32)
    return {
        "width": w,
        "height": h,
        "aspect": w / h,
        "file_kb": size_bytes / 1024,
        "bytes_per_pixel": size_bytes / (w * h),
        "format": fmt,
        "mode": mode,
        "mean": float(a.mean()),
        "std": float(a.std()),
        "frac_black": float((a < 10).mean()),
        "frac_white": float((a > 245).mean()),
    }


def collect_metadata(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as pool:
        meta = list(pool.map(image_metadata, df["path"]))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(meta)], axis=1)


def metadata_features(meta: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(meta[CATEGORICAL_FEATURES].astype(str), dtype=float)
    return pd.concat([meta[NUMERIC_FEATURES].astype(float), dummies], axis=1)


def metadata_shortcut_test(meta: pd.DataFrame, classes: list[str], seed: int = 42) -> dict:
    """Train a random forest on metadata only (train+val rows), test on the test rows.

    A macro AUC near 0.5 means metadata says nothing about the label.
    A macro AUC near 1.0 means the label can be read off the file itself.
    """
    X = metadata_features(meta)
    y = meta["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    fit = meta["split"].isin(["train", "val"]).to_numpy()
    test = (meta["split"] == "test").to_numpy()

    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    clf.fit(X[fit], y[fit])
    probs = np.zeros((test.sum(), len(classes)))
    probs[:, clf.classes_] = clf.predict_proba(X[test])

    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "metrics": compute_metrics(y[test], probs, classes),
        "feature_importance": importance.head(10).round(4).to_dict(),
        "per_class_summary": per_class_summary(meta),
    }


def per_class_summary(meta: pd.DataFrame) -> dict:
    """Median size, brightness and the share of each file format / colour mode per class."""
    out = {}
    for label, g in meta.groupby("label"):
        out[label] = {
            "n": int(len(g)),
            "median_width": float(g.width.median()),
            "median_height": float(g.height.median()),
            "median_aspect": round(float(g.aspect.median()), 3),
            "median_file_kb": round(float(g.file_kb.median()), 1),
            "mean_intensity": round(float(g["mean"].mean()), 1),
            "format_share": g.format.value_counts(normalize=True).round(3).to_dict(),
            "mode_share": g["mode"].value_counts(normalize=True).round(3).to_dict(),
        }
    return out


# ---------------------------------------------------------------- duplicates


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: str | Path, hash_size: int = 16) -> np.ndarray:
    """Difference hash: hash_size**2 bits, packed into uint8.

    Robust to resizing and re-compression, so it finds the same X-ray saved twice
    at different sizes or formats.
    """
    with Image.open(path) as im:
        g = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    return np.packbits(bits.ravel())


def hamming_distances(a: np.ndarray, b: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Pairwise bit differences between packed hashes a (N, B) and b (M, B) -> (N, M)."""
    out = np.empty((len(a), len(b)), dtype=np.uint16)
    for start in range(0, len(a), chunk):
        x = a[start : start + chunk, None, :] ^ b[None, :, :]
        out[start : start + chunk] = _POPCOUNT[x].sum(axis=-1, dtype=np.uint16)
    return out


def duplicate_report(
    df: pd.DataFrame, max_distance: int = 6, hash_size: int = 16, workers: int = 8
) -> tuple[dict, pd.DataFrame]:
    """Find exact (same bytes) and near (dhash within ``max_distance`` bits)
    duplicates. Only test-vs-train/val pairs are listed as near duplicates,
    since those are the ones that inflate test scores.

    Near duplicates are candidates to inspect by eye, not proof: different
    patients can have very similar X-rays.
    """
    df = df.reset_index(drop=True)
    with ThreadPoolExecutor(workers) as pool:
        md5s = list(pool.map(file_md5, df["path"]))
        hashes = np.stack(list(pool.map(lambda p: dhash(p, hash_size), df["path"])))
    df = df.assign(md5=md5s)

    # Exact duplicates
    groups = [g for _, g in df.groupby("md5") if len(g) > 1]
    exact_cross = sum(1 for g in groups if "test" in set(g.split) and set(g.split) - {"test"})
    exact_conflict = sum(1 for g in groups if g.label.nunique() > 1)
    exact_files = sum(len(g) - 1 for g in groups)

    # Near duplicates: test vs train/val
    test_idx = np.flatnonzero(df.split == "test")
    fit_idx = np.flatnonzero(df.split != "test")
    pairs = []
    if len(test_idx) and len(fit_idx):
        dist = hamming_distances(hashes[test_idx], hashes[fit_idx])
        ti, fi = np.nonzero(dist <= max_distance)
        for a, b in zip(ti, fi):
            t, f = test_idx[a], fit_idx[b]
            pairs.append(
                {
                    "test_path": df.path[t],
                    "test_label": df.label[t],
                    "train_path": df.path[f],
                    "train_label": df.label[f],
                    "train_split": df.split[f],
                    "distance": int(dist[a, b]),
                    "exact": bool(df.md5[t] == df.md5[f]),
                }
            )
    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "test_path", "test_label", "train_path", "train_label",
            "train_split", "distance", "exact",
        ],
    ).sort_values(["distance", "test_path"], ignore_index=True)

    summary = {
        "exact_duplicate_groups": len(groups),
        "exact_duplicate_extra_files": exact_files,
        "exact_groups_spanning_train_and_test": exact_cross,
        "exact_groups_with_conflicting_labels": exact_conflict,
        "near_duplicate_max_distance_bits": max_distance,
        "near_duplicate_hash_bits": hash_size * hash_size,
        "test_images_with_near_duplicate_in_train": int(pairs_df.test_path.nunique()),
        "test_images_total": int(len(test_idx)),
        "near_duplicate_pairs": int(len(pairs_df)),
        "near_duplicate_pairs_label_mismatch": int((pairs_df.test_label != pairs_df.train_label).sum()),
    }
    return summary, pairs_df


def save_pair_sheet(pairs: pd.DataFrame, out_path: str | Path, n: int = 12) -> None:
    """Save the closest test/train pairs side by side for a visual check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pairs.head(n)
    if pairs.empty:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)), squeeze=False)
    for row, (_, p) in zip(axes, pairs.iterrows()):
        for ax, key, title in ((row[0], "test", "test"), (row[1], "train", p.train_split)):
            with Image.open(p[f"{key}_path"]) as im:
                ax.imshow(im.convert("L"), cmap="gray")
            ax.set_title(f"{title}: {p[f'{key}_label']}", fontsize=8)
            ax.axis("off")
        row[0].text(0, -0.08, f"distance {p.distance} bits", transform=row[0].transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
