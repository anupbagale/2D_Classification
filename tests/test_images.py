import numpy as np
import pytest
from PIL import Image
import keras

from cxrshortcut.images import (
    DEFAULT_LUNG_BOX,
    build_augmenter,
    load_image,
    lung_box_mask,
    make_dataset,
    occlude_lungs,
)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_load_image_any_mode_gives_3_equal_channels(tmp_path, mode):
    path = tmp_path / f"x_{mode}.png"
    Image.new(mode, (50, 30), color=(120,) * len(mode)).save(path)
    img = load_image(str(path), 32).numpy()
    assert img.shape == (32, 32, 3) and img.dtype == np.float32
    assert 0 <= img.min() and img.max() <= 255
    np.testing.assert_allclose(img[..., 0], img[..., 2])


def test_lung_box_mask_and_occlusion():
    mask = lung_box_mask(100, 100, (0.1, 0.9, 0.2, 0.8))
    assert mask.sum() == 80 * 60
    assert mask[0, 0] == 0 and mask[50, 50] == 1
    img = np.full((100, 100, 3), 200.0, dtype=np.float32)
    out = occlude_lungs(img, (0.1, 0.9, 0.2, 0.8))
    assert out[50, 50].max() == 0 and out[0, 0].min() == 200
    batch = occlude_lungs(img[None], DEFAULT_LUNG_BOX)
    assert batch.shape == (1, 100, 100, 3)


def test_invalid_box():
    with pytest.raises(ValueError):
        lung_box_mask(10, 10, (0.5, 0.4, 0, 1))


def test_augmenter_never_flips():
    aug = build_augmenter(seed=0)
    assert not any(isinstance(layer, keras.layers.RandomFlip) for layer in aug.layers)


def test_dataset_shapes_order_and_occlusion(synth_df):
    classes = ["COVID19", "NORMAL", "PNEUMONIA"]
    test = synth_df[synth_df.split == "test"]
    ds = make_dataset(test, classes, size=32, batch_size=8, training=False)
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds])
    y = np.concatenate(ys)
    assert np.concatenate(xs).shape == (len(test), 32, 32, 3)
    # Evaluation order matches the dataframe order
    expected = test.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    np.testing.assert_array_equal(y.argmax(1), expected)

    occluded = next(iter(make_dataset(test, classes, size=32, batch_size=8, occlude=True)))[0].numpy()
    assert occluded[:, 16, 16, :].max() == 0


def test_training_dataset_augments_within_range(synth_df):
    train = synth_df[synth_df.split == "train"]
    x, y = next(iter(make_dataset(train, ["COVID19", "NORMAL", "PNEUMONIA"], size=32, batch_size=16, training=True)))
    assert x.shape == (16, 32, 32, 3) and y.shape == (16, 3)
    assert float(x.numpy().min()) >= 0 and float(x.numpy().max()) <= 255


def test_unknown_label_raises(synth_df):
    with pytest.raises(ValueError):
        make_dataset(synth_df, ["COVID19", "NORMAL"], size=32)
