"""Convolutional autoencoder over normalized character images.

The encoder maps a 64x64 glyph to a compact latent vector and the decoder
rebuilds the image from it. Training reconstructs both domains of the pair
dataset (degraded and clean glyphs), so latents describe a manifold where
ugly and pretty handwriting live close together and can be interpolated.

The latent blend is what powers `LatentStylizer`: instead of cross-fading
pixels (double exposure), alpha interpolates between the two encodings and
the decoder renders a coherent intermediate handwriting.
"""
import logging
import random
from typing import List

import numpy as np
import torch
from torch import nn

DEFAULT_SIZE = 64


class CharEncoder(nn.Module):
    """Convolutional encoder: (1, size, size) image -> z_dim latent vector."""

    def __init__(self, z_dim: int = 128, size: int = DEFAULT_SIZE):
        """_summary_
        :param z_dim: latent vector length
        :type z_dim: int
        :param size: input canvas side in pixels
        :type size: int
        """
        super().__init__()
        self.z_dim = z_dim
        self.size = size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128 * (size // 8) * (size // 8), z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class CharDecoder(nn.Module):
    """Transposed-convolution decoder: z_dim latent vector -> (1, size, size)."""

    def __init__(self, z_dim: int = 128, size: int = DEFAULT_SIZE):
        """_summary_
        :param z_dim: latent vector length
        :type z_dim: int
        :param size: output canvas side in pixels
        :type size: int
        """
        super().__init__()
        self.z_dim = z_dim
        self.size = size
        self.fc = nn.Linear(z_dim, 128 * (size // 8) * (size // 8))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.fc(z)
        grid = flat.view(-1, 128, self.size // 8, self.size // 8)
        return self.up(grid)


class CharAutoEncoder(nn.Module):
    """Encoder plus decoder sharing one checkpoint."""

    def __init__(self, z_dim: int = 128, size: int = DEFAULT_SIZE):
        """_summary_
        :param z_dim: latent vector length
        :type z_dim: int
        :param size: canvas side in pixels
        :type size: int
        """
        super().__init__()
        self.encoder = CharEncoder(z_dim, size)
        self.decoder = CharDecoder(z_dim, size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def save_checkpoint(model: CharAutoEncoder, path: str) -> None:
    """Persist the autoencoder weights and metadata to disk.

    :param model: trained autoencoder
    :type model: CharAutoEncoder
    :param path: destination file (.pt)
    :type path: str
    """
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "z_dim": model.encoder.z_dim,
            "size": model.encoder.size,
        },
        path,
    )
    logging.info("Checkpoint saved at %s", path)


def load_autoencoder(path: str) -> CharAutoEncoder:
    """Restore an autoencoder saved with save_checkpoint in eval mode.

    :param path: checkpoint file written by train_autoencoder.py
    :type path: str
    :return: the model ready for inference
    :rtype: CharAutoEncoder
    """
    checkpoint = torch.load(path, map_location="cpu")
    model = CharAutoEncoder(z_dim=int(checkpoint["z_dim"]), size=int(checkpoint["size"]))
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    return model


def _collect_samples(data_dir: str, variants_per_glyph: int, seed: int) -> List[np.ndarray]:
    """Gather every training image as float tensors in [0, 1].

    Each clean alphabet glyph contributes itself plus several degraded
    variants, so the autoencoder covers both the pretty and ugly domains.

    :param data_dir: alphabet dataset directory (one subdirectory per class)
    :type data_dir: str
    :param variants_per_glyph: degraded variants per glyph
    :type variants_per_glyph: int
    :param seed: random seed for reproducibility
    :type seed: int
    :return: list of (size, size) float arrays in [0, 1]
    :rtype: List[np.ndarray]
    """
    import cv2
    from pathlib import Path

    from call_ai_grapher.dataset.builder import normalize
    from call_ai_grapher.models.pair_generator import PairGenerator

    generator = PairGenerator(seed=seed)
    samples: List[np.ndarray] = []
    paths = sorted(Path(data_dir).rglob("*.png")) + sorted(Path(data_dir).rglob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No glyph images found under {data_dir}")
    for path in paths:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        clean = normalize(raw, DEFAULT_SIZE).astype(np.float32) / 255.0
        samples.append(clean)
        for _ in range(variants_per_glyph):
            degraded, _ = generator.make_pair(raw)
            samples.append(degraded.astype(np.float32))
    random.Random(seed).shuffle(samples)
    return samples


def train_autoencoder(
    data_dir: str,
    out_path: str,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    variants_per_glyph: int = 10,
    seed: int = 0,
):
    """Train the convolutional autoencoder on the alphabet dataset.

    Reconstruction loss only (L1 on both clean and degraded glyphs); no
    adversarial term is needed because the decoder never has to invent a
    style, just rebuild images from interpolated latents.

    :param data_dir: alphabet dataset directory (one subdirectory per class)
    :type data_dir: str
    :param out_path: where to write the checkpoint
    :type out_path: str
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: samples per optimization step
    :type batch_size: int
    :param lr: learning rate
    :type lr: float
    :param variants_per_glyph: degraded variants generated per glyph
    :type variants_per_glyph: int
    :param seed: random seed
    :type seed: int
    :return: final metrics dict with the reconstruction L1
    :rtype: dict
    """
    torch.manual_seed(seed)
    samples = _collect_samples(data_dir, variants_per_glyph, seed)
    batch = torch.from_numpy(np.stack(samples)).unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(batch), batch_size=batch_size, shuffle=True)

    model = CharAutoEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    model.train()
    loss_value = 0.0
    for epoch in range(epochs):
        epoch_loss, batches = 0.0, 0
        for (images,) in loader:
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        loss_value = epoch_loss / max(batches, 1)
        logging.info("epoch %d/%d - recon L1 %.4f", epoch + 1, epochs, loss_value)

    save_checkpoint(model.eval(), out_path)
    return {"recon_l1": loss_value}
