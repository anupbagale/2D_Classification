import numpy as np
from PIL import Image

from cxrshortcut.audit import (
    collect_metadata,
    dhash,
    duplicate_report,
    hamming_distances,
    metadata_shortcut_test,
)


def test_metadata_detects_planted_shortcut(synth_df):
    meta = collect_metadata(synth_df, workers=2)
    assert {"width", "format", "mode", "mean"} <= set(meta.columns)
    result = metadata_shortcut_test(meta, ["COVID19", "NORMAL", "PNEUMONIA"], seed=0)
    # Synthetic COVID images are RGB PNGs of a different size: metadata gives them away
    assert result["metrics"]["per_class"]["COVID19"]["auc_one_vs_rest"] > 0.95
    assert result["per_class_summary"]["COVID19"]["format_share"] == {"PNG": 1.0}


def test_dhash_survives_resize_and_format_change(tmp_path):
    rng = np.random.default_rng(0)
    a = Image.fromarray((rng.random((64, 64)) * 255).astype(np.uint8))
    a.save(tmp_path / "a.png")
    a.resize((128, 128)).save(tmp_path / "a_big.jpg", quality=95)
    Image.fromarray((rng.random((64, 64)) * 255).astype(np.uint8)).save(tmp_path / "b.png")
    h = np.stack([dhash(tmp_path / n) for n in ("a.png", "a_big.jpg", "b.png")])
    d = hamming_distances(h, h)
    assert d[0, 0] == 0
    assert d[0, 1] <= 20  # same image, resized and re-compressed
    assert d[0, 2] > 80  # unrelated noise images differ in roughly half of 256 bits


def test_hamming_distances_known_values():
    a = np.array([[0b00000000], [0b11111111]], dtype=np.uint8)
    b = np.array([[0b00001111]], dtype=np.uint8)
    np.testing.assert_array_equal(hamming_distances(a, b), [[4], [4]])


def test_duplicate_report_finds_planted_duplicate(synth_df):
    summary, pairs = duplicate_report(synth_df, max_distance=0, workers=2)
    assert summary["exact_groups_spanning_train_and_test"] == 1
    assert summary["exact_groups_with_conflicting_labels"] == 0
    assert pairs.exact.any()
    assert "dup_" in pairs[pairs.exact].test_path.iloc[0]
