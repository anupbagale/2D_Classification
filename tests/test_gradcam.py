import numpy as np
import pytest

from cxrshortcut.gradcam import box_area_fraction, gradcam, lung_fraction
from cxrshortcut.models import build_model


def test_lung_fraction_reference_values():
    box = (0.25, 0.75, 0.25, 0.75)
    uniform = np.ones((1, 40, 40))
    assert lung_fraction(uniform, box)[0] == pytest.approx(box_area_fraction(box, 40)) == pytest.approx(0.25)

    corner = np.zeros((1, 40, 40))
    corner[0, :5, :5] = 1
    centre = np.zeros((1, 40, 40))
    centre[0, 18:22, 18:22] = 1
    assert lung_fraction(corner, box)[0] == 0
    assert lung_fraction(centre, box)[0] == 1
    assert np.isnan(lung_fraction(np.zeros((1, 40, 40)), box)[0])


@pytest.mark.parametrize("name", ["original_cnn", "small_cnn", "densenet121"])
def test_gradcam_shape_and_range(name):
    model = build_model(name, img_size=64, n_classes=3, weights=None)
    x = np.random.default_rng(0).uniform(0, 255, (3, 64, 64, 3)).astype("float32")
    cams = gradcam(model, x)
    assert cams.shape == (3, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    cams_class0 = gradcam(model, x, class_indices=[0, 0, 0])
    assert cams_class0.shape == (3, 64, 64)
