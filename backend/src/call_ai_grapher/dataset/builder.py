"""Alphabet dataset builder.

Generates the reference alphabet ("pretty" handwriting) by rendering every
character of the Spanish alphabet with TrueType fonts, and ingests real
handwriting samples provided by the user. All images are normalized to a
fixed size so later stages (classifier, stylizer) can consume them directly.
"""

import glob as _glob
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import cv2
import numpy as np
from call_ai_grapher.dataset.charset import DEFAULT_ALPHABET
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSION = ".png"


class AlphabetDatasetBuilder:
    def __init__(self, chars: str = DEFAULT_ALPHABET, size: int = 64):
        """_summary_
        :param chars: character classes of the dataset
        :type chars: str
        :param size: normalized image side in pixels
        :type size: int
        """
        self.chars = chars
        self.size = size

    def build_from_fonts(
        self, fonts: Iterable[Union[str, Path, ImageFont.FreeTypeFont]], out_dir: Union[str, Path]
    ) -> Dict[str, int]:
        """Render every character with every font into the dataset.

        :param fonts: font file paths or already loaded fonts
        :type fonts: Iterable[Union[str, Path, ImageFont.FreeTypeFont]]
        :param out_dir: dataset root; images land in `out_dir/<char>/`
        :type out_dir: Union[str, Path]
        :return: number of images stored per class
        :rtype: Dict[str, int]
        """
        counts: Dict[str, List[Path]] = {char: [] for char in self.chars}
        for font in fonts:
            if isinstance(font, ImageFont.FreeTypeFont):
                name = "default"
            else:
                font_path = Path(font)
                try:
                    font = ImageFont.truetype(str(font_path), 96)
                except OSError:
                    logging.warning("Skipping unreadable font %s", font_path)
                    continue
                name = font_path.stem
                name = font_path.stem
            for char in self.chars:
                rendered = render_character(char, font)
                normalized = normalize(rendered, self.size)
                file_name = f"{name}_{ord(char)}{IMAGE_EXTENSION}"
                path = Path(out_dir) / char / file_name
                path.parent.mkdir(parents=True, exist_ok=True)
                _save(path, normalized)
                counts[char].append(path)
        return {char: len(paths) for char, paths in counts.items()}

    def ingest_samples(self, samples_dir: Union[str, Path], out_dir: Union[str, Path]) -> Dict[str, int]:
        """Ingest real handwriting crops organized as `samples_dir/<char>/*`.

        Samples are normalized the same way as the rendered ones so both
        sources are interchangeable for training.

        :param samples_dir: directory with one subdirectory per class
        :type samples_dir: Union[str, Path]
        :param out_dir: dataset root; images land in `out_dir/<char>/`
        :type out_dir: Union[str, Path]
        :return: number of images stored per class
        :rtype: Dict[str, int]
        """
        counts: Dict[str, List[Path]] = {char: [] for char in self.chars}
        samples_dir = Path(samples_dir)
        for class_dir in sorted(p for p in samples_dir.iterdir() if p.is_dir()):
            char = class_dir.name
            if char not in counts:
                logging.warning("Ignoring unknown class directory %s", class_dir)
                continue
            for i, sample in enumerate(sorted(class_dir.glob("*"))):
                image = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logging.warning("Skipping unreadable sample %s", sample)
                    continue
                normalized = normalize(image, self.size)
                path = Path(out_dir) / char / f"sample_{i:04d}{IMAGE_EXTENSION}"
                path.parent.mkdir(parents=True, exist_ok=True)
                _save(path, normalized)
                counts[char].append(path)
        return {char: len(paths) for char, paths in counts.items()}


def render_character(char: str, font: Union[str, Path, ImageFont.FreeTypeFont], padding: int = 8) -> Image.Image:
    """Render a single character on a white canvas.

    :param char: character to render
    :type char: str
    :param font: font file path or already loaded font
    :type font: Union[str, Path, ImageFont.FreeTypeFont]
    :param padding: white margin around the glyph
    :type padding: int
    :return: rendered character image
    :rtype: Image.Image
    """
    if isinstance(font, (str, Path)):
        font = ImageFont.truetype(str(font), 96)
    probe = Image.new("L", (1, 1))
    _, _, w, h = ImageDraw.Draw(probe).textbbox((0, 0), char, font=font)
    image = Image.new("L", (w + padding * 2, h + padding * 2), color=255)
    ImageDraw.Draw(image).text((padding, padding), char, fill=0, font=font)
    return image


def normalize(image: Union[np.ndarray, Image.Image], size: int, content_ratio: float = 0.8) -> np.ndarray:
    """Normalize a character image to a fixed square canvas.

    Steps: grayscale, Otsu binarization, crop to the ink bounding box,
    scale keeping aspect ratio and center on a white canvas.

    :param image: input character image
    :type image: Union[np.ndarray, Image.Image]
    :param size: output canvas side in pixels
    :type size: int
    :param content_ratio: fraction of the canvas the ink may span
    :type content_ratio: float
    :return: normalized grayscale image of shape (size, size)
    :rtype: np.ndarray
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.nonzero(binary)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    if len(xs) == 0:
        return canvas

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    ink = gray[y0:y1, x0:x1]

    target = int(size * content_ratio)
    scale = target / max(ink.shape)
    resized = cv2.resize(ink, (max(int(ink.shape[1] * scale), 1), max(int(ink.shape[0] * scale), 1)))

    dy = (size - resized.shape[0]) // 2
    dx = (size - resized.shape[1]) // 2
    canvas[dy : dy + resized.shape[0], dx : dx + resized.shape[1]] = resized
    return canvas


def _save(path: Path, image: np.ndarray) -> None:
    """Write a grayscale image to disk.

    :param path: destination file path
    :type path: Path
    :param image: grayscale image
    :type image: np.ndarray
    """
    Image.fromarray(image).save(path)


def find_fonts(pattern: str = "fonts/**/*.ttf") -> List[Path]:
    """Find font files matching a glob pattern.

    :param pattern: glob pattern, relative or absolute
    :type pattern: str
    :return: sorted list of matching font paths
    :rtype: List[Path]
    """
    return sorted(Path(p) for p in _glob.glob(pattern))


def class_counts(dataset_dir: Union[str, Path]) -> Dict[str, int]:
    """Count the images stored per class directory.

    :param dataset_dir: dataset root containing one directory per class
    :type dataset_dir: Union[str, Path]
    :return: mapping class -> number of images
    :rtype: Dict[str, int]
    """
    root = Path(dataset_dir)
    return {d.name: len(list(d.glob(f"*{IMAGE_EXTENSION}"))) for d in sorted(root.iterdir()) if d.is_dir()}
