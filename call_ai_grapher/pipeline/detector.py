"""Character detection stage.

Detects character bounding boxes in a scanned page. The default detector
reuses the MSER approach from experiment 7 (object_detection/selective_search)
but returns structured boxes in reading order instead of an annotated image.
Once trained, a YOLO model will back this same interface (ft/char-detector).
"""

import logging
from typing import List

import cv2
import numpy as np

from call_ai_grapher.pipeline.types import CharBox


class CharacterDetector:
    def __init__(self, scale: float = 2.0, min_area: int = 20, max_ratio: float = 2.5):
        """_summary_
        :param scale: upscale applied before running MSER
        :type scale: float
        :param min_area: minimum bounding box area in source pixels
        :type min_area: int
        :param max_ratio: maximum width/height aspect ratio to keep
        :type max_ratio: float
        """
        self.scale = scale
        self.min_area = min_area
        self.max_ratio = max_ratio

    def detect(self, page: np.ndarray) -> List[CharBox]:
        """Return the character boxes found in `page`, in reading order.

        :param page: page image as a numpy array (H, W) or (H, W, 3)
        :type page: np.ndarray
        :return: list of CharBox sorted top-to-bottom, left-to-right
        :rtype: List[CharBox]
        """
        gray = self._to_gray(page)
        scaled = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        regions, _ = cv2.MSER_create().detectRegions(scaled)

        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            box = CharBox(
                x=int(x / self.scale),
                y=int(y / self.scale),
                width=int(w / self.scale),
                height=int(h / self.scale),
            )
            if not self._is_plausible(box):
                continue
            boxes.append(box)

        boxes = self._merge_duplicates(boxes)
        boxes.sort(key=lambda b: (b.y, b.x))
        logging.info("Detected %d characters", len(boxes))
        return boxes

    def _is_plausible(self, box: CharBox) -> bool:
        """Filter out regions that cannot be single characters.

        :param box: candidate box
        :type box: CharBox
        :return: True when the box is plausible
        :rtype: bool
        """
        area = box.width * box.height
        ratio = box.width / max(box.height, 1)
        return area >= self.min_area and ratio <= self.max_ratio

    @staticmethod
    def _merge_duplicates(boxes: List[CharBox]) -> List[CharBox]:
        """Merge heavily overlapping boxes produced by nested MSER regions.

        :param boxes: candidate boxes
        :type boxes: List[CharBox]
        :return: boxes without significant overlaps
        :rtype: List[CharBox]
        """
        merged: List[CharBox] = []
        for box in boxes:
            contained = any(
                other.x <= box.x and other.y <= box.y and other.x2 >= box.x2 and other.y2 >= box.y2 for other in merged
            )
            if contained:
                continue
            merged = [other for other in merged if not _contains(box, other)]
            merged.append(box)
        return merged

    @staticmethod
    def _to_gray(page: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale when needed.

        :param page: input page
        :type page: np.ndarray
        :return: grayscale page
        :rtype: np.ndarray
        """
        if page.ndim == 3:
            return cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        return page


def _contains(outer: CharBox, inner: CharBox) -> bool:
    """Check whether `outer` fully encloses `inner`.

    :param outer: potential container box
    :type outer: CharBox
    :param inner: potential contained box
    :type inner: CharBox
    :return: True when inner is inside outer
    :rtype: bool
    """
    return outer.x <= inner.x and outer.y <= inner.y and outer.x2 >= inner.x2 and outer.y2 >= inner.y2
