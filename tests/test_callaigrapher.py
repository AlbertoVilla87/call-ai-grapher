from pathlib import Path

import numpy as np

from call_ai_grapher import CallAIgraher
from call_ai_grapher.pipeline.types import CharBox


class _FakeDetector:
    def detect(self, page):
        return [CharBox(x=5, y=5, width=10, height=10), CharBox(x=30, y=5, width=10, height=10)]


def _write_synthetic_page(path: Path) -> None:
    import cv2

    page = np.full((50, 50), 255, dtype=np.uint8)
    cv2.circle(page, (10, 10), 4, 0, -1)
    cv2.circle(page, (35, 10), 4, 0, -1)
    cv2.imwrite(str(path), page)


def test_improve_document_end_to_end(tmp_path):
    input_path = tmp_path / "page.png"
    output_path = tmp_path / "out" / "improved.png"
    _write_synthetic_page(input_path)

    grapher = CallAIgraher(detector=_FakeDetector())
    result = grapher.improve_document(str(input_path), str(output_path), alpha=1.0)

    assert output_path.exists()
    assert len(result.chars) == 2
    assert all(char.styled is not None for char in result.chars)


def test_improve_document_alpha_zero_keeps_original(tmp_path):
    input_path = tmp_path / "page.png"
    output_path = tmp_path / "improved.png"
    _write_synthetic_page(input_path)

    grapher = CallAIgraher(detector=_FakeDetector())
    result = grapher.improve_document(str(input_path), str(output_path), alpha=0.0)

    original = result.chars[0].original
    styled = result.chars[0].styled
    assert (original == styled).all()


def test_load_page_missing_file_raises():
    grapher = CallAIgraher()
    try:
        grapher.load_page("no_such_page.png")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
