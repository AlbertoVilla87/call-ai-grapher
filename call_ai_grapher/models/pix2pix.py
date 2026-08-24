"""Pix2pix-lite conditional GAN for per-character style transfer.

A UNet generator maps an "ugly" character crop to its "pretty" counterpart;
a PatchGAN discriminator judges realism of (input, output) stacks. Trained
on the aligned pairs produced by PairDataset. One shared network handles all
characters: the input crop carries the letter shape, the network only learns
the style mapping.
"""

import logging
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from call_ai_grapher.models.pair_generator import PairDataset


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """_summary_
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(nn.Conv2d(32, out_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translate a batch of ugly characters towards the pretty style.

        :param x: batch of images (B, in_channels, H, W) in [0, 1]
        :type x: torch.Tensor
        :return: generated images with the same spatial size as `x`
        :rtype: torch.Tensor
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bottleneck = self.bottleneck(d3)
        u3 = self.up3(bottleneck)
        u2 = self.up2(torch.cat((u3, d3), dim=1))
        u1 = self.up1(torch.cat((u2, d2), dim=1))
        u0 = self.up0(torch.cat((u1, d1), dim=1))
        return self.final(u0)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2):
        """_summary_
        :param in_channels: input + generated channels stacked together
        :type in_channels: int
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Score how real each (source, target) patch looks.

        :param source: input images
        :type source: torch.Tensor
        :param target: candidate outputs
        :type target: torch.Tensor
        :return: patch score map
        :rtype: torch.Tensor
        """
        return self.net(torch.cat((source, target), dim=1))


def save_checkpoint(path: Union[str, Path], generator: UNetGenerator, discriminator: PatchDiscriminator) -> None:
    """Persist generator and discriminator weights.

    :param path: destination checkpoint path
    :type path: Union[str, Path]
    :param generator: trained generator
    :type generator: UNetGenerator
    :param discriminator: trained discriminator
    :type discriminator: PatchDiscriminator
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict()}, path)
    logging.info("Stylizer checkpoint saved to %s", path)


def load_generator(path: Union[str, Path]) -> UNetGenerator:
    """Rebuild a generator from a checkpoint saved by `save_checkpoint`.

    :param path: checkpoint path
    :type path: Union[str, Path]
    :return: generator in evaluation mode
    :rtype: UNetGenerator
    """
    checkpoint = torch.load(str(path), map_location="cpu")
    generator = UNetGenerator()
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    return generator


def train_stylizer(
    data_dir: Union[str, Path],
    out_path: Union[str, Path],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 2e-4,
    variants_per_glyph: int = 20,
    seed: int = 0,
) -> dict:
    """Train the pix2pix stylizer on aligned pairs from an alphabet dataset.

    The loss combines L1 reconstruction (weight 100) with standard GAN
    adversarial loss, following the original pix2pix recipe.

    :param data_dir: alphabet dataset root (one directory per class)
    :type data_dir: Union[str, Path]
    :param out_path: where to write the checkpoint
    :type out_path: Union[str, Path]
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: pairs per optimization step
    :type batch_size: int
    :param lr: learning rate for both networks
    :type lr: float
    :param variants_per_glyph: degraded variants generated per glyph
    :type variants_per_glyph: int
    :param seed: random seed
    :type seed: int
    :return: final metrics {gen_l1, gen_gan, disc}
    :rtype: dict
    """
    torch.manual_seed(seed)
    dataset = PairDataset(data_dir, variants_per_glyph=variants_per_glyph)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator, discriminator = UNetGenerator(), PatchDiscriminator()
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        total_l1, total_gan, total_disc, seen = 0.0, 0.0, 0.0, 0
        for ugly, pretty in loader:
            real_label = torch.ones(len(ugly), 1, *discriminator(ugly, pretty).shape[2:])
            fake_label = torch.zeros_like(real_label)

            optimizer_d.zero_grad()
            pred = generator(ugly).detach()
            disc_loss = (
                criterion_gan(discriminator(ugly, pretty), real_label)
                + criterion_gan(discriminator(ugly, pred), fake_label)
            ) / 2
            disc_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            pred = generator(ugly)
            gan_loss = criterion_gan(discriminator(ugly, pred), real_label)
            l1_loss = criterion_l1(pred, pretty)
            gen_loss = gan_loss + 100.0 * l1_loss
            gen_loss.backward()
            optimizer_g.step()

            total_l1 += l1_loss.item() * len(ugly)
            total_gan += gan_loss.item() * len(ugly)
            total_disc += disc_loss.item() * len(ugly)
            seen += len(ugly)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logging.info(
                "epoch %d/%d - L1 %.4f - gan %.4f - disc %.4f",
                epoch + 1,
                epochs,
                total_l1 / seen,
                total_gan / seen,
                total_disc / seen,
            )

    metrics = {"gen_l1": total_l1 / seen, "gen_gan": total_gan / seen, "disc": total_disc / seen}
    save_checkpoint(out_path, generator, discriminator)
    return metrics
