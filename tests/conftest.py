import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from cxrshortcut.data import add_validation_split, index_dataset
from cxrshortcut.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    """Small synthetic dataset with a planted COVID shortcut and one train/test duplicate."""
    root = tmp_path_factory.mktemp("synth")
    return make_synthetic_dataset(root, n_train=20, n_test=8, size=64, seed=0)


@pytest.fixture(scope="session")
def synth_df(synth_root):
    return add_validation_split(index_dataset(synth_root), val_fraction=0.25, seed=1)
