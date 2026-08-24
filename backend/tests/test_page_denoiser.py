import cv2
import numpy as np
from call_ai_grapher.pipeline.page_denoiser import PageDenoiser


def _shadowed_noisy_page() -> np.ndarray:
    page = np.full((120, 200, 3), 240, dtype=np.float32)
    gradient = np.linspace(0.35, 1.0, 200, dtype=np.float32)
    page *= gradient[np.newaxis, :, np.newaxis]
    page = page.astype(np.uint8)
    cv2.circle(page, (100, 60), 6, 0, -1)
    rng = np.random.default_rng(7)
    for _ in range(40):
        x, y = int(rng.integers(5, 195)), int(rng.integers(5, 115))
        page[y, x] = 0
    return page


def test_clean_flattens_illumination():
    page = _shadowed_noisy_page()

    cleaned = PageDenoiser().clean(page)

    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    shadowed_side = gray[:, :20].mean()
    lit_side = gray[:, -20:].mean()
    assert abs(shadowed_side - lit_side) < 30


def test_clean_removes_specks_and_keeps_stroke():
    page = _shadowed_noisy_page()

    cleaned = PageDenoiser(min_speck_area=12).clean(page)

    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    areas = stats[1:, cv2.CC_STAT_AREA]
    assert (areas > 12).sum() >= 1
    assert (areas < 12).sum() == 0


def test_clean_preserves_shape_and_dtype():
    page = _shadowed_noisy_page()

    cleaned = PageDenoiser().clean(page)

    assert cleaned.shape == page.shape
    assert cleaned.dtype == page.dtype
