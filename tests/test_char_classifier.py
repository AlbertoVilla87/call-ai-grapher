import numpy as np
import pytest
import torch

from call_ai_grapher.models import CharCNN, load_checkpoint, save_checkpoint, train_model
from call_ai_grapher.pipeline.classifier import CharClassifier
from call_ai_grapher.pipeline.types import CharBox


@pytest.fixture(scope="module")
def alphabet_dir(tmp_path_factory):
    """A tiny 3-class alphabet dataset with easily separable glyphs."""
    import cv2

    root = tmp_path_factory.mktemp("alphabet")
    for char in ("A", "B", "Ñ"):
        class_dir = root / char
        class_dir.mkdir()
        for i in range(4):
            glyph = np.full((64, 64), 255, dtype=np.uint8)
            if char == "A":
                cv2.circle(glyph, (32, 32), 14 + i, 0, -1)
            elif char == "B":
                cv2.line(glyph, (32, 10), (32, 54), 0, 4 + i)
            else:
                cv2.line(glyph, (12, 12), (52, 52), 0, 4 + i)
            cv2.imwrite(str(class_dir / f"g{i}.png"), glyph)
    return root


def test_char_cnn_forward_shape():
    model = CharCNN(num_classes=27)
    out = model(torch.rand(4, 1, 64, 64))
    assert out.shape == (4, 27)


def test_train_model_reaches_perfect_accuracy_on_synthetic(alphabet_dir, tmp_path):
    metrics = train_model(alphabet_dir, tmp_path / "classifier.pt", epochs=30, seed=0)
    assert metrics["accuracy"] > 0.9


def test_checkpoint_roundtrip(alphabet_dir, tmp_path):
    train_model(alphabet_dir, tmp_path / "classifier.pt", epochs=1, seed=0)
    model, classes = load_checkpoint(tmp_path / "classifier.pt")
    assert classes == ["A", "B", "Ñ"]
    assert not model.training


def test_classifier_labels_boxes(alphabet_dir, tmp_path):
    train_model(alphabet_dir, tmp_path / "classifier.pt", epochs=30, seed=0)

    import cv2

    page = np.full((100, 200), 255, dtype=np.uint8)
    glyph = cv2.imread(str(alphabet_dir / "A" / "g0.png"), cv2.IMREAD_GRAYSCALE)
    page[10:74, 20:84] = glyph
    boxes = [CharBox(x=20, y=10, width=64, height=64)]

    classifier = CharClassifier(str(tmp_path / "classifier.pt"))
    labeled = classifier.classify(page, boxes)

    assert labeled[0].label == "A"
    assert labeled[0].confidence > 0.5


def test_classify_crop_returns_valid_label_and_confidence(tmp_path):
    classifier = CharClassifier(_train_mini(tmp_path))
    label, confidence = classifier.classify_crop(np.full((32, 32), 255, dtype=np.uint8))
    assert label in ("A", "B", "Ñ")
    assert 0.0 <= confidence <= 1.0


def _mini_dataset(root):
    import cv2

    data_dir = root / "mini"
    for char in ("A", "B", "Ñ"):
        class_dir = data_dir / char
        class_dir.mkdir(parents=True)
        for i in range(2):
            glyph = np.full((64, 64), 255, dtype=np.uint8)
            cv2.rectangle(glyph, (12 + i, 12 + i), (50, 50), 0, 2)
            cv2.imwrite(str(class_dir / f"g{i}.png"), glyph)
    return str(data_dir)


def _train_mini(root):
    from call_ai_grapher.models import train_model

    checkpoint = root / "mini.pt"
    train_model(_mini_dataset(root), checkpoint, epochs=1, seed=0)
    return str(checkpoint)
