import numpy as np
import pytest
import torch
from call_ai_grapher.models import (
    PairDataset,
    PairGenerator,
    PatchDiscriminator,
    UNetGenerator,
    train_stylizer,
)
from call_ai_grapher.pipeline.stylizer import NeuralStylizer


@pytest.fixture(scope="module")
def alphabet_dir(tmp_path_factory):
    """A tiny 3-class alphabet dataset."""
    import cv2

    root = tmp_path_factory.mktemp("alphabet")
    for i, char in enumerate(("A", "B", "Ñ")):
        class_dir = root / char
        class_dir.mkdir()
        for j in range(3):
            glyph = np.full((64, 64), 255, dtype=np.uint8)
            if char == "A":
                cv2.circle(glyph, (32, 32), 12 + j * 2 + i, 0, -1)
            elif char == "B":
                cv2.line(glyph, (30 + j, 10), (30 + j, 54), 0, 4)
            else:
                cv2.line(glyph, (12, 12), (52, 52), 0, 4 + j)
            cv2.imwrite(str(class_dir / f"g{j}.png"), glyph)
    return root


def test_pair_generation_is_aligned(alphabet_dir):
    import cv2

    glyph = cv2.imread(str(alphabet_dir / "A" / "g0.png"), cv2.IMREAD_GRAYSCALE)
    degraded, clean = PairGenerator(seed=1).make_pair(glyph)

    assert degraded.shape == clean.shape == (64, 64)
    assert degraded.max() <= 1.0 and degraded.min() >= 0.0
    assert not np.allclose(degraded, clean)
    ink_clean = (clean < 0.5).sum()
    ink_degraded = (degraded < 0.5).sum()
    assert ink_degraded > 0.2 * ink_clean


def test_pair_dataset_length_and_item(alphabet_dir):
    dataset = PairDataset(alphabet_dir, variants_per_glyph=4, seed=0)
    assert len(dataset) == 9 * 4
    ugly, pretty = dataset[0]
    assert ugly.shape == pretty.shape == (1, 64, 64)


def test_unet_generator_preserves_size():
    generator = UNetGenerator()
    out = generator(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_patch_discriminator_output():
    discriminator = PatchDiscriminator()
    score = discriminator(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
    assert score.shape[0] == 2 and score.shape[1] == 1


def test_train_stylizer_improves_reconstruction(alphabet_dir, tmp_path):
    checkpoint = tmp_path / "stylizer.pt"
    train_stylizer(alphabet_dir, checkpoint, epochs=1, batch_size=8, variants_per_glyph=3, seed=0)

    from call_ai_grapher.models import load_generator

    generator = load_generator(checkpoint)
    with torch.no_grad():
        before = generator(torch.rand(1, 1, 64, 64))
    train_stylizer(alphabet_dir, checkpoint, epochs=20, batch_size=8, variants_per_glyph=3, seed=0)
    generator_after = load_generator(checkpoint)
    with torch.no_grad():
        after = generator_after(torch.rand(1, 1, 64, 64))
    assert (after - before).abs().mean() > 0.01


def test_neural_stylizer_alpha_regulator(alphabet_dir, tmp_path):
    checkpoint = tmp_path / "stylizer.pt"
    train_stylizer(alphabet_dir, checkpoint, epochs=2, batch_size=8, variants_per_glyph=2, seed=0)
    stylizer = NeuralStylizer(str(checkpoint))

    crop = np.full((40, 40, 3), 200, dtype=np.uint8)

    import cv2

    untouched = stylizer.stylize(crop, alpha=0.0)
    assert (untouched == cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).all()

    low = stylizer.stylize(crop, alpha=0.01).astype(int)
    high = stylizer.stylize(crop, alpha=1.0).astype(int)
    mid = stylizer.stylize(crop, alpha=0.5).astype(int)

    assert low.shape == high.shape == mid.shape == (64, 64)
    assert not np.allclose(low, high)
    # the regulator is an exact cross-fade: stylize(0.5) == mean(stylize(0.25), stylize(0.75))
    expected = 0.5 * stylizer.stylize(crop, alpha=0.25).astype(float) + 0.5 * stylizer.stylize(crop, alpha=0.75).astype(
        float
    )
    assert np.abs(mid - expected).mean() < 2.0
