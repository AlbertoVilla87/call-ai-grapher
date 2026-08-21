import numpy as np

from call_ai_grapher.dataset import (
    DEFAULT_ALPHABET,
    SPANISH_ALPHABET_LOWER,
    SPANISH_ALPHABET_UPPER,
    AlphabetDatasetBuilder,
    normalize,
    render_character,
)


def test_spanish_alphabets_have_27_classes_with_enye():
    assert len(SPANISH_ALPHABET_UPPER) == 27
    assert "Ñ" in SPANISH_ALPHABET_UPPER
    assert len(SPANISH_ALPHABET_LOWER) == 27
    assert "ñ" in SPANISH_ALPHABET_LOWER
    assert DEFAULT_ALPHABET == SPANISH_ALPHABET_UPPER


def test_render_character_produces_ink():
    from PIL import ImageFont

    font = ImageFont.load_default(size=48)
    image = render_character("A", font)
    arr = np.array(image)
    assert image.mode == "L"
    assert (arr < 128).sum() > 0


def test_normalize_returns_square_centered_canvas():
    image = np.full((40, 90), 255, dtype=np.uint8)
    image[10:30, 20:70] = 0
    normalized = normalize(image, size=64)
    assert normalized.shape == (64, 64)

    ys, xs = np.nonzero(normalized < 128)
    center_x = (xs.min() + xs.max()) / 2
    center_y = (ys.min() + ys.max()) / 2
    assert abs(center_x - 31.5) < 3
    assert abs(center_y - 31.5) < 3


def test_normalize_empty_image_returns_white_canvas():
    image = np.full((32, 32), 255, dtype=np.uint8)
    normalized = normalize(image, size=64)
    assert normalized.shape == (64, 64)
    assert (normalized == 255).all()


def test_build_from_fonts_creates_class_directories(tmp_path):
    from PIL import ImageFont

    font = ImageFont.load_default(size=48)
    builder = AlphabetDatasetBuilder(size=64)
    counts = builder.build_from_fonts([font], tmp_path)

    assert len(counts) == 27
    assert all(count == 1 for count in counts.values())
    assert (tmp_path / "Ñ").is_dir()
    assert list((tmp_path / "A").glob("*.png"))


def test_ingest_samples_normalizes_and_stores(tmp_path):
    from PIL import Image

    samples = tmp_path / "samples" / "B"
    samples.mkdir(parents=True)
    for i, width in enumerate((20, 35)):
        image = np.full((50, width), 255, dtype=np.uint8)
        image[15:35, 5 : width - 5] = 0
        Image.fromarray(image).save(samples / f"{i}.png")

    builder = AlphabetDatasetBuilder(size=64)
    counts = builder.ingest_samples(tmp_path / "samples", tmp_path / "dataset")

    assert counts["B"] == 2
    first_stored = sorted((tmp_path / "dataset" / "B").glob("*.png"))[0]
    assert np.array(Image.open(first_stored)).shape == (64, 64)


def test_ingest_samples_ignores_unknown_classes(tmp_path):
    unknown = tmp_path / "samples" / "1"
    unknown.mkdir(parents=True)
    builder = AlphabetDatasetBuilder(size=64)
    counts = builder.ingest_samples(tmp_path / "samples", tmp_path / "dataset")
    assert sum(counts.values()) == 0
