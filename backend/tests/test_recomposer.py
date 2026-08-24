import numpy as np
from call_ai_grapher.pipeline.recomposer import Recomposer, estimate_baseline
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


def test_estimate_baseline_returns_lowest_ink_row():
    crop = np.full((20, 20), 255, dtype=np.uint8)
    crop[5:13, 6:14] = 0
    assert estimate_baseline(crop) == 12


def test_estimate_baseline_blank_crop_returns_none():
    assert estimate_baseline(np.full((20, 20), 255, dtype=np.uint8)) is None


def test_compose_aligns_replacement_to_source_baseline():
    page = _page()
    box = CharBox(x=10, y=10, width=20, height=20)
    original = np.full((20, 20), 255, dtype=np.uint8)
    original[2:7, 6:14] = 0
    styled = np.full((20, 20), 255, dtype=np.uint8)
    styled[8:17, 6:14] = 0

    result = Recomposer().compose(page, [StyledChar(box=box, original=original, styled=styled)])

    assert (result[16, 16:24] < 50).all()
    assert (result[24:27, 16:24] > 200).all()


def test_compose_records_estimated_baseline_in_box():
    page = _page()
    box = CharBox(x=10, y=10, width=20, height=20)
    original = np.full((20, 20), 255, dtype=np.uint8)
    original[2:7, 6:14] = 0
    styled = np.zeros((20, 20), dtype=np.uint8)

    Recomposer().compose(page, [StyledChar(box=box, original=original, styled=styled)])

    assert box.baseline == 6


def test_compose_falls_back_when_source_has_no_ink():
    page = _page()
    box = CharBox(x=10, y=10, width=20, height=20)
    original = np.full((20, 20), 255, dtype=np.uint8)
    styled = np.full((20, 20), 255, dtype=np.uint8)
    styled[0:5, 6:14] = 0

    result = Recomposer().compose(page, [StyledChar(box=box, original=original, styled=styled)])

    assert (result[10:15, 16:24] < 50).all()
