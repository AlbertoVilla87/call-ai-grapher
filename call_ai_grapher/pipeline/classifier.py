"""Character classification stage.

Assigns a label (A-Z plus ñ) and a confidence to each detected character
crop. Backed by a small CNN trained on the alphabet dataset built by
build_alphabet.py (see ft/char-classifier).
"""
import logging
from typing import List, Tuple

import numpy as np

from call_ai_grapher.pipeline.types import CharBox


class CharClassifier:
    def classify(self, page: np.ndarray, boxes: List[CharBox]) -> List[CharBox]:
        """Fill in `label` and `confidence` for each box, in place.

        :param page: page image the boxes refer to
        :type page: np.ndarray
        :param boxes: boxes returned by the detector
        :type boxes: List[CharBox]
        :return: the same boxes with labels assigned
        :rtype: List[CharBox]
        """
        raise NotImplementedError("CharClassifier is implemented in ft/char-classifier")

    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a single character crop.

        :param crop: grayscale character image
        :type crop: np.ndarray
        :return: (label, confidence)
        :rtype: Tuple[str, float]
        """
        raise NotImplementedError("CharClassifier is implemented in ft/char-classifier")
