import pytest

from cxrshortcut.data import (
    add_validation_split,
    class_names,
    counts_table,
    find_split_dirs,
    index_dataset,
)


def test_finds_nested_train_and_test(synth_root):
    train, test = find_split_dirs(synth_root.parent)  # search from one level up
    assert train.name == "train" and test.name == "test"


def test_index_counts(synth_root):
    df = index_dataset(synth_root)
    assert set(df.split) == {"train", "test"}
    assert class_names(df) == ["COVID19", "NORMAL", "PNEUMONIA"]
    assert (df[df.split == "train"].label.value_counts() == 20).all()
    # 8 test images per class plus one planted duplicate in COVID19
    assert df[df.split == "test"].label.value_counts().to_dict() == {"COVID19": 9, "NORMAL": 8, "PNEUMONIA": 8}


def test_class_filter(synth_root):
    df = index_dataset(synth_root, classes=["COVID19", "NORMAL"])
    assert class_names(df) == ["COVID19", "NORMAL"]
    with pytest.raises(ValueError):
        index_dataset(synth_root, classes=["COVID19", "TYPO"])


def test_validation_split_is_stratified_disjoint_and_reproducible(synth_root):
    df = index_dataset(synth_root)
    a = add_validation_split(df, val_fraction=0.25, seed=3)
    b = add_validation_split(df, val_fraction=0.25, seed=3)
    assert a.equals(b)
    # Test rows are never touched
    assert (a[df.split == "test"].split == "test").all()
    # Each image is in exactly one split, and every class is in val
    assert a.path.is_unique
    counts = counts_table(a)
    assert (counts["val"] == 5).all() and (counts["train"] == 15).all()


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_split_dirs(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        find_split_dirs(tmp_path)
