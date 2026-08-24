"""Character classification stage.

Assigns a label (A-Z plus Ñ) and a confidence to each detected character
crop using the CharCNN trained by `train_classifier.py` on the alphabet
dataset built by `build_alphabet.py`.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch

from call_ai_grapher.dataset.builder import normalize
from call_ai_grapher.models.char_cnn import CharCNN, load_checkpoint
from call_ai_grapher.pipeline.types import CharBox


class CharClassifier:
    def __init__(self, model_path: str = "models/char_classifier.pt"):
        """_summary_
        :param model_path: path to the checkpoint written by train_classifier.py
        :type model_path: str
        """
        self.model, self.classes = load_checkpoint(model_path)

    def classify(self, page: np.ndarray, boxes: List[CharBox]) -> List[CharBox]:
        """Fill in `label` and `confidence` for each box, in place.

        :param page: page image the boxes refer to
        :type page: np.ndarray
        :param boxes: boxes returned by the detector
        :type boxes: List[CharBox]
        :return: the same boxes with labels assigned
        :rtype: List[CharBox]
        """
        for box in boxes:
            label, confidence = self.classify_crop(box.crop_from(page))
            box.label = label
            box.confidence = confidence
        logging.info("Classified %d characters", len(boxes))
        return boxes

    def classify_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        """Classify a single character crop.

        The crop is normalized exactly like the training data, so raw
        detections from any detector are handled uniformly.

        :param crop: character image (grayscale or BGR)
        :type crop: np.ndarray
        :return: (label, confidence)
        :rtype: Tuple[str, float]
        """
        image = normalize(crop, self.model.size).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, index = probabilities.max(0)
        return self.classes[int(index)], float(confidence)
