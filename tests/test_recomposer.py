import numpy as np

from call_ai_grapher.pipeline.recomposer import Recomposer
from call_ai_grapher.pipeline.types import CharBox, StyledChar


def _page() -> np.ndarray:
    return np.full((100, 100), 255, dtype=np.uint8)


def test_compose_pastes_styled_characters():
    page = _page()
    box = CharBox(x=10, y=10, width=20, height=20)
    styled = np.zeros((20, 20), dtype=np.uint8)
    result = Recomposer().compose(page, [StyledChar(box=box, original=page[10:30, 10:30], styled=styled)])
    assert (result[10:30, 10:30] == 0).all()
    assert (result[30:, :] == 255).all()


def test_compose_skips_chars_without_replacement():
    page = _page()
    box = CharBox(x=10, y=10, width=20, height=20)
    result = Recomposer().compose(page, [StyledChar(box=box, original=page[10:30, 10:30], styled=None)])
    assert (result == 255).all()


def test_compose_resizes_mismatched_patch():
    page = _page()
    box = CharBox(x=0, y=0, width=10, height=10)
    styled = np.zeros((5, 5), dtype=np.uint8)
    result = Recomposer().compose(page, [StyledChar(box=box, original=page[0:10, 0:10], styled=styled)])
    assert (result[0:10, 0:10] == 0).all()
