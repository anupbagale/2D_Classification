import numpy as np
import pytest

from cxrshortcut.models import (
    DENSENET_FINE_TUNE_FROM,
    build_model,
    last_conv_layer_name,
    set_backbone_trainable,
)
from cxrshortcut.train import class_weights


def n_trainable(model):
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


@pytest.mark.parametrize("name", ["original_cnn", "small_cnn", "densenet121"])
def test_models_output_probabilities(name):
    model = build_model(name, img_size=64, n_classes=3, weights=None)
    x = np.random.default_rng(0).uniform(0, 255, (2, 64, 64, 3)).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p.sum(1), 1, rtol=1e-5)
    assert len(model.get_layer(last_conv_layer_name(model)).output.shape) == 4


def test_original_cnn_has_its_flatten_dense_bottleneck():
    model = build_model("original_cnn", img_size=256, n_classes=2)
    dense = [layer for layer in model.layers if layer.name.startswith("dense")][0]
    # v1 notebook: 7,372,864 parameters in this one layer
    assert dense.count_params() == 7_372_864


def test_densenet_starts_frozen_and_unfreezes_last_block():
    model = build_model("densenet121", img_size=64, n_classes=3, weights=None)
    head_only = n_trainable(model)
    assert head_only == 1024 * 3 + 3  # just the new Dense layer
    set_backbone_trainable(model, DENSENET_FINE_TUNE_FROM)
    assert head_only < n_trainable(model) < model.count_params()
    assert not any(layer.trainable for layer in model.layers if "bn" in layer.name)
    with pytest.raises(ValueError):
        set_backbone_trainable(model, "no_such_layer")


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("resnet9000", 64, 3)


def test_class_weights_balance_classes():
    import pandas as pd

    labels = pd.Series(["A"] * 10 + ["B"] * 30)
    w = class_weights(labels, ["A", "B"])
    assert w[0] == pytest.approx(2.0) and w[1] == pytest.approx(2 / 3)
    assert w[0] * 10 == pytest.approx(w[1] * 30)
