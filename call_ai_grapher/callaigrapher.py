import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from call_ai_grapher.pipeline.classifier import CharClassifier
from call_ai_grapher.pipeline.detector import CharacterDetector
from call_ai_grapher.pipeline.recomposer import Recomposer
from call_ai_grapher.pipeline.stylizer import Stylizer
from call_ai_grapher.pipeline.types import DocumentResult, StyledChar


class CallAIgraher:
    """Orchestrates the handwriting improvement pipeline over a scanned page."""

    def __init__(
        self,
        detector: Optional[CharacterDetector] = None,
        classifier: Optional[CharClassifier] = None,
        stylizer: Optional[Stylizer] = None,
        recomposer: Optional[Recomposer] = None,
    ):
        """_summary_
        :param detector: character detection stage, defaults to MSER detector
        :type detector: Optional[CharacterDetector]
        :param classifier: character classification stage, defaults to None
        :type classifier: Optional[CharClassifier]
        :param stylizer: character stylization stage, defaults to baseline blend
        :type stylizer: Optional[Stylizer]
        :param recomposer: page recomposition stage, defaults to paste back
        :type recomposer: Optional[Recomposer]
        """
        self.detector = detector or CharacterDetector()
        self.classifier = classifier
        self.stylizer = stylizer or Stylizer()
        self.recomposer = recomposer or Recomposer()

    def improve_document(self, input_path: str, output_path: str, alpha: float = 1.0) -> DocumentResult:
        """Improve the handwriting of a scanned document.

        :param input_path: path to the scanned page image
        :type input_path: str
        :param output_path: path where the improved page is saved
        :type output_path: str
        :param alpha: improvement amount in [0, 1]; 0 keeps the original
            stroke, 1 applies the full pretty style
        :type alpha: float
        :return: pipeline result with boxes, styled chars and output path
        :rtype: DocumentResult
        """
        page = self.load_page(input_path)
        result = DocumentResult(source_path=Path(input_path))

        result.boxes = self.detector.detect(page)
        if self.classifier is not None:
            try:
                result.boxes = self.classifier.classify(page, result.boxes)
            except NotImplementedError:
                logging.warning("Classifier not implemented yet; skipping labels")

        result.chars = [
            StyledChar(
                box=box,
                original=box.crop_from(page),
                styled=self.stylizer.stylize(box.crop_from(page), alpha, label=box.label),
            )
            for box in result.boxes
        ]

        improved = self.recomposer.compose(page, result.chars)
        result.output_path = Path(output_path)
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(result.output_path), improved)
        logging.info("Improved document saved to %s", result.output_path)
        return result

    @staticmethod
    def load_page(path: str) -> np.ndarray:
        """Load a page image from disk.

        :param path: image file path
        :type path: str
        :return: page image as a numpy array
        :rtype: np.ndarray
        """
        page = cv2.imread(path)
        if page is None:
            raise FileNotFoundError(f"Cannot read image at {path}")
        return page
