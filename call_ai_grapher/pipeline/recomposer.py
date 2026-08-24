"""Document recomposition stage.

Pastes the styled characters back into the page at their original positions,
aligning replacements vertically by character baseline so strokes sit on the
same line instead of floating inside their bounding boxes (see
ft/document-recomposer).
"""

import logging
from typing import List, Optional

import numpy as np

from call_ai_grapher.pipeline.types import CharBox, StyledChar


class Recomposer:
    def compose(self, page: np.ndarray, chars: List[StyledChar]) -> np.ndarray:
        """Return a new page where every styled character replaces its original.

        Characters without a styled replacement are left untouched. When both
        the original crop and its replacement have detectable ink, the patch
        is shifted vertically so both baselines coincide; otherwise the patch
        simply fills the original box.

        :param page: original page image
        :type page: np.ndarray
        :param chars: characters with their styled replacements
        :type chars: List[StyledChar]
        :return: the improved page image
        :rtype: np.ndarray
        """
        result = page.copy()
        replaced = 0
        for char in chars:
            if char.styled is None:
                continue
            box = char.box
            patch = char.styled
            if patch.ndim == 2 and result.ndim == 3:
                patch = np.stack((patch,) * 3, axis=-1)
            elif patch.ndim == 3 and result.ndim == 2:
                result = np.stack((result,) * 3, axis=-1)
            region = result[box.y : box.y2, box.x : box.x2]
            patch = _fit(patch, region.shape)

            src_baseline = box.baseline if box.baseline is not None else estimate_baseline(char.original)
            dst_baseline = estimate_baseline(patch)
            if box.baseline is None:
                box.baseline = src_baseline

            if src_baseline is None or dst_baseline is None:
                result[box.y : box.y2, box.x : box.x2] = patch
            else:
                _paste_aligned(result, patch, box, src_baseline - dst_baseline)
            replaced += 1
        logging.info("Replaced %d of %d characters", replaced, len(chars))
        return result


def estimate_baseline(crop: np.ndarray) -> Optional[int]:
    """Return the row index of the lowest ink pixel inside `crop`.

    Ink is separated with an Otsu threshold after removing speckle noise, so
    scans with uneven lighting are handled. Crops without a clear foreground
    (blank patches or uniform regions) have no measurable baseline.

    :param crop: character image, grayscale or BGR
    :type crop: np.ndarray
    :return: baseline row index, or None when no ink is detected
    :rtype: Optional[int]
    """
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.size == 0 or int(gray.std()) < 2:
        return None
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 3)
    rows = np.where(binary.any(axis=1))[0]
    return int(rows[-1]) if len(rows) else None


def _paste_aligned(page: np.ndarray, patch: np.ndarray, box: CharBox, shift: int) -> None:
    """Clear `box` on `page` and paste `patch` shifted vertically by `shift`.

    Rows pushed outside the page are clipped. The exposed area is filled with
    the local background estimated from the original box pixels.

    :param page: page being recomposed, modified in place
    :type page: np.ndarray
    :param patch: styled patch already sized to the box
    :type patch: np.ndarray
    :param box: source box of the character being replaced
    :type box: CharBox
    :param shift: vertical offset applied to the patch, positive moves down
    :type shift: int
    """
    region = page[box.y : box.y2, box.x : box.x2]
    background = np.median(region, axis=(0, 1)).astype(page.dtype)
    page[box.y : box.y2, box.x : box.x2] = background

    height = patch.shape[0]
    src_start = max(0, -(box.y + shift))
    src_end = height - max(0, box.y + shift + height - page.shape[0])
    if src_end <= src_start:
        page[box.y : box.y2, box.x : box.x2] = patch
        return
    page[box.y + shift + src_start : box.y + shift + src_end, box.x : box.x2] = patch[src_start:src_end]


def _fit(patch: np.ndarray, shape: tuple) -> np.ndarray:
    """Resize `patch` to `shape` when it does not match already.

    :param patch: image patch to paste
    :type patch: np.ndarray
    :param shape: target shape (rows, cols[, channels])
    :type shape: tuple
    :return: the patch adjusted to the target shape
    :rtype: np.ndarray
    """
    import cv2

    if patch.shape == shape:
        return patch
    return cv2.resize(patch, (shape[1], shape[0]))
