"""YOLO-backed character detection stage.

Wraps an ultralytics YOLO model trained to locate characters on a page
(single class). Implements the same interface as the MSER detector, so the
pipeline orchestrator can swap between them transparently.
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

from call_ai_grapher.pipeline.types import CharBox


class YoloCharacterDetector:
    def __init__(self, model_path: Union[str, Path] = "models/character_detector.pt", confidence: float = 0.25):
        """_summary_
        :param model_path: path to the trained YOLO weights
        :type model_path: Union[str, Path]
        :param confidence: minimum detection confidence
        :type confidence: float
        """
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, page: np.ndarray) -> List[CharBox]:
        """Return the character boxes found in `page`, in reading order.

        :param page: page image as a numpy array (H, W) or (H, W, 3)
        :type page: np.ndarray
        :return: list of CharBox sorted top-to-bottom, left-to-right
        :rtype: List[CharBox]
        """
        results = self.model.predict(page, conf=self.confidence, verbose=False)
        boxes: List[CharBox] = []
        for result in results:
            for xyxy in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = xyxy
                boxes.append(CharBox(x=int(x1), y=int(y1), width=int(x2 - x1), height=int(y2 - y1)))
        boxes.sort(key=lambda b: (b.y, b.x))
        logging.info("Detected %d characters", len(boxes))
        return boxes
