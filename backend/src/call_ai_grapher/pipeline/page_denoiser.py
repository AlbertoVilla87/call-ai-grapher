"""Whole-page preprocessing stage.

Phone photos of paper show strong shadows and gray backgrounds, and the
per-character stylizer only touches detected crops, so anything outside
them reaches the output untouched. This optional stage normalizes the
page before detection: dividing each channel by a heavily blurred
estimate of the background flattens illumination adaptively, and small
dark connected components (specks) are removed.
"""

import logging

import cv2
import numpy as np


class PageDenoiser:
    def __init__(self, kernel_size: int = 51, min_speck_area: int = 12):
        """_summary_
        :param kernel_size: background blur window; larger values keep thicker strokes intact
        :type kernel_size: int
        :param min_speck_area: dark connected components below this area in px are dropped
        :type min_speck_area: int
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.min_speck_area = min_speck_area

    def __call__(self, page: np.ndarray) -> np.ndarray:
        return self.clean(page)

    def clean(self, page: np.ndarray) -> np.ndarray:
        """Return a copy of `page` with flattened illumination and specks removed.

        :param page: BGR or grayscale page image
        :type page: np.ndarray
        :return: cleaned page with the same shape and dtype
        :rtype: np.ndarray
        """
        cleaned = self._flatten_illumination(page)
        return self._despeckle(cleaned)

    def _flatten_illumination(self, page: np.ndarray) -> np.ndarray:
        """Divide each channel by its blurred background estimate.

        The background estimate approximates the local paper color, so
        shadows and gray tints divide out towards white while strokes stay
        dark. The denominator is clamped so deep shadows cannot blow up
        the division.

        :param page: input image, uint8 BGR or grayscale
        :type page: np.ndarray
        :return: image with an even, white background
        :rtype: np.ndarray
        """
        k = self._fitting_kernel(min(page.shape[:2]))
        background = cv2.GaussianBlur(page.astype(np.float32), (k, k), 0)
        normalized = page.astype(np.float32) / np.maximum(background, 32.0) * 255.0
        return np.clip(normalized, 0, 255).astype(page.dtype)

    def _despeckle(self, page: np.ndarray) -> np.ndarray:
        """Drop dark connected components smaller than `min_speck_area`.

        :param page: BGR or grayscale page image
        :type page: np.ndarray
        :return: page without isolated specks
        :rtype: np.ndarray
        """
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        specks = (stats[1:, cv2.CC_STAT_AREA] < self.min_speck_area).nonzero()[0] + 1
        if specks.size == 0:
            return page
        mask = np.isin(labels, specks)
        cleaned = page.copy()
        cleaned[mask] = 255 if page.ndim == 2 else np.array([255, 255, 255], dtype=page.dtype)
        logging.info("Removed %d specks", specks.size)
        return cleaned

    def _fitting_kernel(self, dim: int) -> int:
        """Return an odd kernel size that fits within `dim` pixels.

        :param dim: smallest image dimension
        :type dim: int
        :return: odd kernel size between 3 and `self.kernel_size`
        :rtype: int
        """
        largest_odd = dim if dim % 2 == 1 else dim - 1
        return max(min(self.kernel_size, largest_odd), 3)
