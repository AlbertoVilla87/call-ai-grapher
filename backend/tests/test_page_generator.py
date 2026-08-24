from pathlib import Path

import numpy as np
import pytest
import yaml
from call_ai_grapher.dataset import CLASS_ID, PageGenerator, write_detector_dataset


@pytest.fixture(scope="module")
def glyph_dir(tmp_path_factory):
    """A tiny alphabet dataset built from synthetic glyphs."""
    import cv2

    root = tmp_path_factory.mktemp("glyphs")
    for char in ("A", "B", "Ñ"):
        class_dir = root / char
        class_dir.mkdir()
        for i in range(2):
            glyph = np.full((64, 64), 255, dtype=np.uint8)
            cv2.rectangle(glyph, (16, 16), (48, 48 + i * 4), 0, 2)
            cv2.imwrite(str(class_dir / f"g{i}.png"), glyph)
    return root


def test_page_generator_produces_page_and_valid_labels(glyph_dir):
    generator = PageGenerator(glyph_dir, page_size=(320, 240), seed=1)
    page, labels = generator.generate_page()

    assert page.shape == (240, 320, 3)
    assert len(labels) > 10
    for class_id, cx, cy, w, h in labels:
        assert class_id == CLASS_ID
        assert 0 <= cx <= 1 and 0 <= cy <= 1
        assert 0 < w < 1 and 0 < h < 1


def test_page_generator_is_reproducible_with_seed(glyph_dir):
    page_a, labels_a = PageGenerator(glyph_dir, seed=7).generate_page()
    page_b, labels_b = PageGenerator(glyph_dir, seed=7).generate_page()
    assert labels_a == labels_b
    assert (page_a == page_b).all()


def test_write_detector_dataset_layout(tmp_path, glyph_dir):
    data_yaml = write_detector_dataset(glyph_dir, tmp_path / "detector", train_pages=3, val_pages=2, seed=3)

    assert data_yaml.exists()
    config = yaml.safe_load(data_yaml.read_text())
    assert config["names"] == {0: "character"}
    for split, count in (("train", 3), ("val", 2)):
        images = list((tmp_path / "detector" / "images" / split).glob("*.png"))
        labels = list((tmp_path / "detector" / "labels" / split).glob("*.txt"))
        assert len(images) == count
        assert len(labels) == count
        first_label = labels[0].read_text().strip().splitlines()[0].split()
        assert len(first_label) == 5


def test_yolo_detector_returns_boxes():
    pytest.importorskip("ultralytics")

    from call_ai_grapher.pipeline.yolo_detector import YoloCharacterDetector

    try:
        detector = YoloCharacterDetector("yolov8n.pt")
    except Exception as error:  # offline environment without pretrained weights
        pytest.skip(f"YOLO weights unavailable: {error}")

    page = np.full((240, 320), 255, dtype=np.uint8)
    boxes = detector.detect(page)
    assert isinstance(boxes, list)
    for box in boxes:
        assert box.width > 0 and box.height > 0
