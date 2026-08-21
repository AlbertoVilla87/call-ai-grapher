"""Document recomposition stage.

Pastes the styled characters back into the page at their original positions,
preserving line layout and baseline alignment (see ft/document-recomposer).
"""
import logging
from typing import List

import numpy as np

from call_ai_grapher.pipeline.types import StyledChar


class Recomposer:
    def compose(self, page: np.ndarray, chars: List[StyledChar]) -> np.ndarray:
        """Return a new page where every styled character replaces its original.

        Characters without a styled replacement are left untouched.

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
            result[box.y : box.y2, box.x : box.x2] = _fit(patch, region.shape)
            replaced += 1
        logging.info("Replaced %d of %d characters", replaced, len(chars))
        return result


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
