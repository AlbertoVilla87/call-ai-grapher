"""Aligned (ugly, pretty) pair generation for style transfer training.

Each clean glyph from the alphabet dataset is degraded with handwriting-like
distortions (elastic deformation, slant, thickness changes, noise and blur)
to synthesize its "ugly" counterpart. Because both images derive from the
same render, pairs are pixel-aligned and no annotation is needed.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from call_ai_grapher.dataset.builder import normalize


class PairGenerator:
    def __init__(self, size: int = 64, seed: int = 0):
        """_summary_
        :param size: normalized glyph side in pixels
        :type size: int
        :param seed: random seed for reproducible degradations
        :type seed: int
        """
        self.size = size
        self.rng = np.random.default_rng(seed)

    def make_pair(self, glyph: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create one aligned (ugly, pretty) pair from a clean glyph.

        :param glyph: normalized grayscale glyph
        :type glyph: np.ndarray
        :return: (degraded, clean) float32 tensors of shape (size, size) in [0, 1]
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        clean = normalize(glyph, self.size).astype(np.float32) / 255.0
        degraded = clean.copy()

        degraded = _elastic(degraded, self.rng)
        degraded = _slant(degraded, self.rng)
        degraded = _thickness(degraded, self.rng)
        degraded = _noise_and_blur(degraded, self.rng)

        return degraded.astype(np.float32), clean.astype(np.float32)


class PairDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        variants_per_glyph: int = 20,
        size: int = 64,
        seed: int = 0,
    ):
        """_summary_
        :param data_dir: alphabet dataset root (one directory per class)
        :type data_dir: Union[str, Path]
        :param variants_per_glyph: degraded variants generated per glyph
        :type variants_per_glyph: int
        :param size: normalized glyph side in pixels
        :type size: int
        :param seed: random seed
        :type seed: int
        """
        generator = PairGenerator(size=size, seed=seed)
        self.pairs = []
        for path in sorted(Path(data_dir).glob("*/*.png")):
            glyph = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if glyph is None:
                logging.warning("Skipping unreadable glyph %s", path)
                continue
            for _ in range(variants_per_glyph):
                self.pairs.append(generator.make_pair(glyph))
        logging.info("Pair dataset ready: %d pairs", len(self.pairs))

    def __len__(self) -> int:
        """Return the number of training pairs.

        :return: dataset size
        :rtype: int
        """
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one (ugly, pretty) pair as tensors.

        :param index: pair index
        :type index: int
        :return: two tensors of shape (1, size, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        degraded, clean = self.pairs[index]
        return torch.from_numpy(degraded).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0)


def _elastic(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply elastic deformation to a [0, 1] grayscale image.

    :param image: input image
    :type image: np.ndarray
    :param rng: numpy random generator
    :type rng: np.random.Generator
    :return: deformed image
    :rtype: np.ndarray
    """
    shape = image.shape
    smooth = max(shape[0] // 8, 4)
    dx = cv2.GaussianBlur(rng.normal(0, 2.2, shape).astype(np.float32), (0, 0), smooth)
    dy = cv2.GaussianBlur(rng.normal(0, 2.2, shape).astype(np.float32), (0, 0), smooth)
    x, y = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
    return cv2.remap(image, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)


def _slant(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random shear and slight scale change to the image.

    :param image: input image
    :type image: np.ndarray
    :param rng: numpy random generator
    :type rng: np.random.Generator
    :return: slanted image
    :rtype: np.ndarray
    """
    shear = float(rng.uniform(-0.25, 0.25))
    scale = float(rng.uniform(0.9, 1.05))
    matrix = np.array([[scale, shear, 0], [0, scale, 0]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=1.0)


def _thickness(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly thin or thicken the strokes.

    :param image: input image
    :type image: np.ndarray
    :param rng: numpy random generator
    :type rng: np.random.Generator
    :return: morphologically transformed image
    :rtype: np.ndarray
    """
    ink = (image < 0.5).astype(np.uint8)
    kernel_size = int(rng.integers(1, 3)) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if rng.random() < 0.5:
        ink = cv2.erode(ink, kernel, iterations=1)
    else:
        ink = cv2.dilate(ink, kernel, iterations=1)
    return np.where(ink > 0, 0.0, 1.0).astype(np.float32)


def _noise_and_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add gaussian noise and blur mimicking a bad scan.

    :param image: input image
    :type image: np.ndarray
    :param rng: numpy random generator
    :type rng: np.random.Generator
    :return: degraded image
    :rtype: np.ndarray
    """
    noisy = image + rng.normal(0, float(rng.uniform(0.02, 0.12)), image.shape).astype(np.float32)
    if rng.random() < 0.6:
        noisy = cv2.GaussianBlur(noisy, (3, 3), 0)
    return np.clip(noisy, 0.0, 1.0)
