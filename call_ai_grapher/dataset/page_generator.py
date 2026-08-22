"""Synthetic page generation for detector training.

Composes normalized alphabet glyphs into realistic scanned-looking pages,
emitting the image plus character bounding boxes in YOLO label format
(`class_id cx cy w h`, coordinates normalized to [0, 1]). The detector is
single-class: it only locates characters; labeling each one is the
classifier's job.
"""
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import cv2
import numpy as np

CLASS_ID = 0
CLASS_NAME = "character"


class PageGenerator:
    def __init__(
        self,
        glyph_dir: Union[str, Path],
        page_size: Tuple[int, int] = (640, 480),
        seed: int = 0,
    ):
        """_summary_
        :param glyph_dir: normalized alphabet dataset (one directory per class)
        :type glyph_dir: Union[str, Path]
        :param page_size: generated page size as (width, height)
        :type page_size: Tuple[int, int]
        :param seed: random seed for reproducible pages
        :type seed: int
        """
        self.page_width, self.page_height = page_size
        self.rng = np.random.default_rng(seed)
        self.glyphs = self._load_glyphs(Path(glyph_dir))

    def _load_glyphs(self, glyph_dir: Path) -> Dict[str, List[np.ndarray]]:
        """Load every normalized glyph grouped by class.

        :param glyph_dir: dataset root containing one directory per class
        :type glyph_dir: Path
        :return: mapping class name -> list of glyph images
        :rtype: Dict[str, List[np.ndarray]]
        """
        glyphs: Dict[str, List[np.ndarray]] = {}
        for class_dir in sorted(p for p in glyph_dir.iterdir() if p.is_dir()):
            images = []
            for path in sorted(class_dir.glob("*.png")):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
            if images:
                glyphs[class_dir.name] = images
        logging.info("Loaded %d classes with %d glyphs", len(glyphs), sum(len(v) for v in glyphs.values()))
        return glyphs

    @property
    def classes(self) -> List[str]:
        """Sorted class names as used by the generator.

        :return: ordered class names
        :rtype: List[str]
        """
        return sorted(self.glyphs)

    def generate_page(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Generate one synthetic page with its YOLO labels.

        Characters are laid out in lines with random spacing and size, then
        the page is degraded with noise and blur to mimic a scan.

        :return: (page image in BGR, list of (class_id, cx, cy, w, h) normalized)
        :rtype: Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]
        """
        page = np.full((self.page_height, self.page_width), 255, dtype=np.uint8)
        labels: List[Tuple[int, float, float, float, float]] = []

        y = int(self.rng.integers(20, 60))
        while y < self.page_height - 40:
            x = int(self.rng.integers(15, 40))
            while x < self.page_width - 50:
                char = self.classes[self.rng.integers(len(self.classes))]
                glyph = self.glyphs[char][self.rng.integers(len(self.glyphs[char]))]
                height = int(self.rng.integers(22, 44))
                width = max(int(glyph.shape[1] * height / glyph.shape[0]), 6)
                glyph = cv2.resize(glyph, (width, height))

                ink = self.rng.integers(30, 130)
                stamped = glyph.copy()
                stamped[stamped < 128] = ink

                x_end, y_end = x + width, y + height
                if x_end >= self.page_width or y_end >= self.page_height:
                    break
                region = page[y:y_end, x:x_end]
                page[y:y_end, x:x_end] = np.minimum(region, stamped)

                labels.append(
                    (
                        CLASS_ID,
                        (x + width / 2) / self.page_width,
                        (y + height / 2) / self.page_height,
                        width / self.page_width,
                        height / self.page_height,
                    )
                )
                x += width + int(self.rng.integers(6, 18))
            y += int(self.rng.integers(46, 70))

        degraded = self._degrade(page)
        return degraded, labels

    def _degrade(self, page: np.ndarray) -> np.ndarray:
        """Apply scan-like degradation: blur, noise and uneven brightness.

        :param page: clean page image
        :type page: np.ndarray
        :return: degraded page in BGR
        :rtype: np.ndarray
        """
        degraded = page.copy()
        if self.rng.random() < 0.7:
            degraded = cv2.GaussianBlur(degraded, (3, 3), 0)
        noise = self.rng.normal(0, 6, degraded.shape).astype(np.float32)
        degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        gradient = np.linspace(1.0, float(self.rng.uniform(0.75, 0.95)), degraded.shape[1], dtype=np.float32)
        degraded = (degraded.astype(np.float32) * gradient[None, :]).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(degraded, cv2.COLOR_GRAY2BGR)

    def iter_pages(self, count: int) -> Iterator[Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]]:
        """Generate `count` pages.

        :param count: number of pages to generate
        :type count: int
        :yield: (page image, YOLO labels)
        :rtype: Iterator[Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]]
        """
        for _ in range(count):
            yield self.generate_page()


def write_detector_dataset(
    glyph_dir: Union[str, Path],
    out_dir: Union[str, Path],
    train_pages: int = 200,
    val_pages: int = 40,
    seed: int = 0,
) -> Path:
    """Write a YOLO-format detection dataset from alphabet glyphs.

    Layout: `images/{train,val}/*.png`, sibling `labels/{train,val}/*.txt`
    and a `data.yaml` ready for ultralytics.

    :param glyph_dir: normalized alphabet dataset directory
    :type glyph_dir: Union[str, Path]
    :param out_dir: output dataset root
    :type out_dir: Union[str, Path]
    :param train_pages: number of training pages
    :type train_pages: int
    :param val_pages: number of validation pages
    :type val_pages: int
    :param seed: random seed
    :type seed: int
    :return: path to the generated `data.yaml`
    :rtype: Path
    """
    out_dir = Path(out_dir)
    generator = PageGenerator(glyph_dir, seed=seed)
    for split, page_count in (("train", train_pages), ("val", val_pages)):
        images_dir = out_dir / "images" / split
        labels_dir = out_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for i, (page, labels) in enumerate(generator.iter_pages(page_count)):
            stem = f"{split}_{i:04d}"
            cv2.imwrite(str(images_dir / f"{stem}.png"), page)
            with open(labels_dir / f"{stem}.txt", "w") as handle:
                handle.writelines(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in labels)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  {CLASS_ID}: {CLASS_NAME}\n"
    )
    logging.info(
        "Detection dataset written to %s (%d train / %d val pages)", out_dir, train_pages, val_pages
    )
    return data_yaml
