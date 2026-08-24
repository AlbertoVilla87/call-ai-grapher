import numpy as np

import web_ui
from call_ai_grapher.pipeline.factory import build_stylizer
from call_ai_grapher.pipeline.types import CharBox


def _page():
    page = np.full((50, 50), 220, dtype=np.uint8)
    import cv2

    cv2.circle(page, (20, 20), 6, 40, -1)
    return np.stack((page,) * 3, axis=-1)


def test_analyze_page_without_image_returns_hint():
    state, before, status = web_ui.analyze_page(None, "mser", "", 0.25, "", "baseline", "", "", "")

    assert state is None
    assert before is None
    assert "Upload" in status


def test_analyze_page_reports_pipeline_error_in_status(tmp_path):
    missing = str(tmp_path / "no_such_font.pt")

    state, _, status = web_ui.analyze_page(_page(), "yolo", missing, 0.25, "", "baseline", "", "", "")

    assert state is None
    assert "Pipeline error" in status


def test_render_page_returns_none_before_analysis():
    assert web_ui.render_page(None, 0.8) is None


def test_analyze_and_render_roundtrip():
    boxes = [CharBox(x=10, y=10, width=20, height=20)]
    state = {
        "page": _page(),
        "boxes": boxes,
        "stylizer": build_stylizer("baseline"),
    }

    improved = web_ui.render_page(state, 1.0)
    original = web_ui.render_page(state, 0.0)

    assert improved is not None and original is not None
    assert improved.shape == original.shape == (50, 50, 3)
    assert (improved != original).any()


def test_latent_stylizer_requires_classifier():
    try:
        build_stylizer("latent")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
