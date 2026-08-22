"""Character stylization stage.

Improves a character crop towards the target ("pretty") handwriting style.
`alpha` is the improvement regulator requested by the user: 0 keeps the
original stroke, 1 applies the full pretty style.

The baseline implementation blends the crop with its binarized version, so
the pipeline is usable end to end before any model is trained. It will be
replaced by latent-space interpolation on the GAN/autoencoder (see
ft/style-stylizer and ft/blend-regulator) while keeping this same interface.
"""
import logging

import cv2
import numpy as np


class Stylizer:
    def __init__(self, block_size: int = 31, c: int = 15):
        """_summary_
        :param block_size: adaptive threshold neighbourhood size
        :type block_size: int
        :param c: adaptive threshold constant
        :type c: int
        """
        self.block_size = block_size
        self.c = c

    def stylize(self, crop: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Blend the raw crop with its cleaned version.

        :param crop: grayscale character image
        :type crop: np.ndarray
        :param alpha: improvement amount in [0, 1]
        :type alpha: float
        :return: the stylized character image
        :rtype: np.ndarray
        """
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha == 0.0:
            return crop.copy()
        clean = self._clean(crop)
        return ((1.0 - alpha) * crop.astype(np.float32) + alpha * clean.astype(np.float32)).astype(crop.dtype)

    def _clean(self, crop: np.ndarray) -> np.ndarray:
        """Return a binarized version of the crop with the noise removed.

        :param crop: character image, grayscale or BGR
        :type crop: np.ndarray
        :return: cleaned character image with the same channels as `crop`
        :rtype: np.ndarray
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.c
        )
        binary = cv2.medianBlur(binary, 3)
        if crop.ndim == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary


class NeuralStylizer:
    """Style transfer stylizer backed by the pix2pix generator.

    The improvement regulator works as a cross-fade: alpha blends the
    original stroke with the network output, so 0 keeps your handwriting
    and 1 applies the full pretty style.
    """

    def __init__(self, model_path: str = "models/char_stylizer.pt"):
        """_summary_
        :param model_path: checkpoint written by train_stylizer.py
        :type model_path: str
        """
        import torch

        from call_ai_grapher.models.pix2pix import load_generator

        self._torch = torch
        self.generator = load_generator(model_path)

    def stylize(self, crop: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Translate the crop towards the pretty style and cross-fade by alpha.

        :param crop: character image (grayscale or BGR)
        :type crop: np.ndarray
        :param alpha: improvement amount in [0, 1]
        :type alpha: float
        :return: the stylized character image
        :rtype: np.ndarray
        """
        from call_ai_grapher.dataset.builder import normalize

        alpha = min(max(alpha, 0.0), 1.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if alpha == 0.0:
            return gray.copy()

        normalized = normalize(gray, _model_size(self.generator))
        tensor = self._torch.from_numpy(normalized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with self._torch.no_grad():
            generated = self.generator(tensor)[0, 0].numpy()
        blended = ((1.0 - alpha) * (normalized.astype(np.float32) / 255.0) + alpha * generated).clip(0, 1)
        return (blended * 255).astype(np.uint8)


def _model_size(generator) -> int:
    """Infer the expected input size of a trained UNetGenerator.

    The architecture is fully convolutional but checkpoints are trained on
    fixed-size inputs; this keeps parity with dataset normalization.

    :param generator: loaded generator
    :return: input image side in pixels
    """
    return 64
