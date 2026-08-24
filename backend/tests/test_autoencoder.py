import numpy as np
import pytest
import torch
from call_ai_grapher.models import CharAutoEncoder, train_autoencoder
from call_ai_grapher.pipeline.stylizer import LatentStylizer


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


def test_encoder_decoder_shapes():
    model = CharAutoEncoder(z_dim=32)
    x = torch.rand(2, 1, 64, 64)
    z = model.encoder(x)
    assert z.shape == (2, 32)
    rebuilt = model.decoder(z)
    assert rebuilt.shape == x.shape
    assert rebuilt.min() >= 0.0 and rebuilt.max() <= 1.0


def test_training_improves_reconstruction(alphabet_dir, tmp_path):
    import cv2
    from call_ai_grapher.dataset.builder import normalize
    from call_ai_grapher.models import load_autoencoder

    glyph = normalize(cv2.imread(str(alphabet_dir / "A" / "g0.png"), cv2.IMREAD_GRAYSCALE), 64)
    sample = torch.from_numpy(glyph.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    checkpoint = tmp_path / "ae.pt"
    metrics = train_autoencoder(alphabet_dir, checkpoint, epochs=1, batch_size=8, variants_per_glyph=1, seed=0)
    model = load_autoencoder(checkpoint)
    with torch.no_grad():
        before = (model(sample) - sample).abs().mean()
    train_autoencoder(alphabet_dir, checkpoint, epochs=40, batch_size=8, variants_per_glyph=1, seed=0)
    trained = load_autoencoder(checkpoint)
    with torch.no_grad():
        after = (trained(sample) - sample).abs().mean()

    assert metrics["recon_l1"] > 0
    assert after < before


def test_latent_blend_matches_manual_interpolation(alphabet_dir, tmp_path):
    checkpoint = tmp_path / "ae.pt"
    train_autoencoder(alphabet_dir, checkpoint, epochs=5, batch_size=8, variants_per_glyph=2, seed=0)
    stylizer = LatentStylizer(str(checkpoint), alphabet_dir=str(alphabet_dir))

    crop = np.full((40, 40, 3), 200, dtype=np.uint8)

    import cv2
    from call_ai_grapher.dataset.builder import normalize

    untouched = stylizer.stylize(crop, alpha=0.0, label="A")
    assert (untouched == cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).all()

    mid = stylizer.stylize(crop, alpha=0.5, label="A")
    normalized = normalize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 64)
    with torch.no_grad():
        z_crop = stylizer.model.encoder(
            torch.from_numpy(normalized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        )
        expected = stylizer.model.decoder(0.5 * z_crop + 0.5 * stylizer._reference_latent("A"))[0, 0].numpy()
    assert np.abs(mid.astype(float) - expected * 255.0).max() <= 1.0

    # the blend moves monotonically towards the pretty reference as alpha grows
    reference = normalize(cv2.imread(str(alphabet_dir / "A" / "g0.png"), cv2.IMREAD_GRAYSCALE), 64).astype(float)
    near = np.abs(stylizer.stylize(crop, alpha=0.9, label="A").astype(float) - reference).mean()
    far = np.abs(stylizer.stylize(crop, alpha=0.1, label="A").astype(float) - reference).mean()
    assert near < far


def test_latent_blend_reaches_reference(alphabet_dir, tmp_path):
    checkpoint = tmp_path / "ae.pt"
    train_autoencoder(alphabet_dir, checkpoint, epochs=60, batch_size=8, variants_per_glyph=2, seed=0)
    stylizer = LatentStylizer(str(checkpoint), alphabet_dir=str(alphabet_dir))

    crop = np.full((40, 40, 3), 200, dtype=np.uint8)
    full_style = stylizer.stylize(crop, alpha=1.0, label="B")

    import cv2

    reference = cv2.imread(str(alphabet_dir / "B" / "g0.png"), cv2.IMREAD_GRAYSCALE)
    distance_to_reference = np.abs(full_style.astype(float) - reference.astype(float)).mean()
    assert distance_to_reference < 60.0


def test_unknown_label_falls_back_to_own_latent(alphabet_dir, tmp_path):
    checkpoint = tmp_path / "ae.pt"
    train_autoencoder(alphabet_dir, checkpoint, epochs=2, batch_size=8, variants_per_glyph=1, seed=0)
    stylizer = LatentStylizer(str(checkpoint), alphabet_dir=str(alphabet_dir))

    crop = np.full((40, 40, 3), 128, dtype=np.uint8)
    out = stylizer.stylize(crop, alpha=0.8, label=None)
    assert out.shape == (64, 64) and out.dtype == np.uint8
