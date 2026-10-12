"""Models. All take float images in [0, 255] and return softmax probabilities."""

from __future__ import annotations

import keras
from keras import layers

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HEAD_LAYERS = {"head_gap", "head_dropout", "predictions"}
DENSENET_FINE_TUNE_FROM = "conv5_block1_0_bn"


def build_original_cnn(input_shape, n_classes: int) -> keras.Model:
    """The v1 architecture from notebooks/original, kept as a baseline.

    Nearly all its parameters sit in the Flatten -> Dense(64) layer. Built with
    the functional API (not Sequential) so Grad-CAM can reach inner layers.
    """
    inputs = keras.Input(input_shape, name="image")
    x = layers.Rescaling(1 / 255.0)(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, activation="relu", name="last_conv")(x)
    x = layers.MaxPool2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="original_cnn")


def build_small_cnn(input_shape, n_classes: int) -> keras.Model:
    """A small CNN with batch norm and global average pooling (~0.4M parameters)."""
    inputs = keras.Input(input_shape, name="image")
    x = layers.Rescaling(1 / 255.0)(inputs)
    filters = (32, 64, 128, 256)
    for i, f in enumerate(filters):
        last = i == len(filters) - 1
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu", name="last_conv" if last else None)(x)
        if not last:
            x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="small_cnn")


def build_densenet121(input_shape, n_classes: int, weights: str | None = "imagenet") -> keras.Model:
    """ImageNet-pretrained DenseNet121 (the CheXNet backbone) with a new head.

    The backbone starts frozen; see ``set_backbone_trainable`` for fine-tuning.
    """
    inputs = keras.Input(input_shape, name="image")
    x = layers.Rescaling(1 / 255.0)(inputs)
    x = layers.Normalization(
        mean=list(IMAGENET_MEAN), variance=[s**2 for s in IMAGENET_STD], name="imagenet_norm"
    )(x)
    base = keras.applications.DenseNet121(include_top=False, weights=weights, input_tensor=x)
    x = layers.GlobalAveragePooling2D(name="head_gap")(base.output)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="densenet121")
    set_backbone_trainable(model, unfreeze_from=None)
    return model


def set_backbone_trainable(model: keras.Model, unfreeze_from: str | None = None) -> None:
    """Freeze everything except the head; if ``unfreeze_from`` is a layer name,
    also train that layer and all later ones. Batch-norm layers stay frozen,
    which is the usual practice when fine-tuning on a small dataset."""
    names = [layer.name for layer in model.layers]
    if unfreeze_from is not None and unfreeze_from not in names:
        raise ValueError(f"Layer {unfreeze_from!r} not in model")
    unfreeze = False
    for layer in model.layers:
        if layer.name in HEAD_LAYERS:
            layer.trainable = True
            continue
        if layer.name == unfreeze_from:
            unfreeze = True
        layer.trainable = unfreeze and not isinstance(layer, layers.BatchNormalization)


MODEL_BUILDERS = {
    "original_cnn": build_original_cnn,
    "small_cnn": build_small_cnn,
    "densenet121": build_densenet121,
}


def build_model(name: str, img_size: int, n_classes: int, weights: str | None = "imagenet") -> keras.Model:
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_BUILDERS)}")
    shape = (img_size, img_size, 3)
    if name == "densenet121":
        return build_densenet121(shape, n_classes, weights=weights)
    return MODEL_BUILDERS[name](shape, n_classes)


def last_conv_layer_name(model: keras.Model) -> str:
    """The layer Grad-CAM should explain: the last 4-D conv/activation output."""
    for layer in reversed(model.layers):
        if not isinstance(layer, (layers.Conv2D, layers.Activation, layers.ReLU)):
            continue
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found")
