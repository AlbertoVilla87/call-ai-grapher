import numpy as np
import pytest

from call_ai_grapher.pipeline.types import CharBox


def test_box_corners():
    box = CharBox(x=10, y=20, width=5, height=7)
    assert box.x2 == 15
    assert box.y2 == 27


def test_crop_from_returns_enclosed_pixels():
    page = np.zeros((50, 50), dtype=np.uint8)
    page[20:27, 10:15] = 255
    box = CharBox(x=10, y=20, width=5, height=7)
    crop = box.crop_from(page)
    assert crop.shape == (7, 5)
    assert crop.sum() == 255 * 35


@pytest.mark.parametrize("alpha", [-0.5, 2.0])
def test_stylizer_clamps_alpha(alpha):
    from call_ai_grapher.pipeline.stylizer import Stylizer

    crop = np.full((10, 10), 128, dtype=np.uint8)
    stylized = Stylizer().stylize(crop, alpha)
    assert stylized.shape == crop.shape
