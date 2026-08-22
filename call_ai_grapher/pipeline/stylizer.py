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

    def stylize(self, crop: np.ndarray, alpha: float = 1.0, label=None) -> np.ndarray:
        """Blend the raw crop with its cleaned version.

        :param crop: grayscale character image
        :type crop: np.ndarray
        :param alpha: improvement amount in [0, 1]
        :type alpha: float
        :param label: optional character label (unused by this backend)
        :type label: Optional[str]
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

    def stylize(self, crop: np.ndarray, alpha: float = 1.0, label=None) -> np.ndarray:
        """Translate the crop towards the pretty style and cross-fade by alpha.

        :param crop: character image (grayscale or BGR)
        :type crop: np.ndarray
        :param alpha: improvement amount in [0, 1]
        :type alpha: float
        :param label: optional character label (unused by this backend)
        :type label: Optional[str]
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


class LatentStylizer:
    """Regulator via latent interpolation between the crop and its pretty reference.

    Instead of cross-fading pixels (which doubles the exposure at mid
    alphas), both images are encoded and the blend happens in latent space:
    z = (1-alpha) * E(crop) + alpha * E(reference), then D(z) renders a
    coherent intermediate handwriting. The reference pretty glyph is looked
    up by class label inside the alphabet dataset directory.
    """

    def __init__(self, model_path: str = "models/char_autoencoder.pt", alphabet_dir: str = "dataset/alphabet"):
        """_summary_
        :param model_path: checkpoint written by train_autoencoder.py
        :type model_path: str
        :param alphabet_dir: alphabet dataset used to find reference glyphs
        :type alphabet_dir: str
        """
        import torch

        from call_ai_grapher.models.autoencoder import load_autoencoder

        self._torch = torch
        self.model = load_autoencoder(model_path)
        self.alphabet_dir = alphabet_dir
        self._reference_latents = {}

    def stylize(self, crop: np.ndarray, alpha: float = 1.0, label=None) -> np.ndarray:
        """Interpolate the crop encoding towards its class reference by alpha.

        :param crop: character image (grayscale or BGR)
        :type crop: np.ndarray
        :param alpha: improvement amount in [0, 1]
        :type alpha: float
        :param label: character label used to pick the pretty reference glyph
        :type label: Optional[str]
        :return: the stylized character image
        :rtype: np.ndarray
        """
        from call_ai_grapher.dataset.builder import normalize

        torch = self._torch
        alpha = min(max(alpha, 0.0), 1.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if alpha == 0.0:
            return gray.copy()

        normalized = normalize(gray, self.model.encoder.size)
        tensor = torch.from_numpy(normalized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            z_crop = self.model.encoder(tensor)
            reference = self._reference_latent(label)
            if reference is None:
                logging.warning("No reference glyph for label %r; decoding the crop latent only", label)
                return (self.model.decoder(z_crop)[0, 0].numpy().clip(0, 1) * 255).astype(np.uint8)
            z_blend = (1.0 - alpha) * z_crop + alpha * reference
            generated = self.model.decoder(z_blend)[0, 0].numpy()
        return (generated.clip(0, 1) * 255).astype(np.uint8)

    def _reference_latent(self, label):
        """Return the cached latent of the first alphabet glyph for `label`.

        :param label: character class such as "A" or "Ñ"
        :type label: Optional[str]
        :return: latent tensor of shape (1, z_dim), or None when unavailable
        :rtype: Optional[torch.Tensor]
        """
        from pathlib import Path

        if label is None or not self.alphabet_dir:
            return None
        if label in self._reference_latents:
            return self._reference_latents[label]

        class_dir = Path(self.alphabet_dir) / str(label)
        if not class_dir.is_dir():
            logging.warning("Alphabet directory has no class %s", label)
            return None
        glyph_path = next(iter(sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))), None)
        if glyph_path is None:
            logging.warning("Class %s has no reference images", label)
            return None
        import cv2

        from call_ai_grapher.dataset.builder import normalize

        raw = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
        normalized = normalize(raw, self.model.encoder.size)
        tensor = self._torch.from_numpy(normalized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with self._torch.no_grad():
            latent = self.model.encoder(tensor)
        self._reference_latents[label] = latent
        return latent


def _model_size(generator) -> int:
    """Infer the expected input size of a trained UNetGenerator.

    The architecture is fully convolutional but checkpoints are trained on
    fixed-size inputs; this keeps parity with dataset normalization.

    :param generator: loaded generator
    :return: input image side in pixels
    """
    return 64
