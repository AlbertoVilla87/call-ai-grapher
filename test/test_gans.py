import os
import sys

import torch
import unittest
from unittest import TestCase
from unittest.mock import patch, MagicMock

sys.path.append(os.getcwd())

from call_ai_grapher import gans


class TestGetNoise(TestCase):
    def test_get_noise_cpu(self):
        n_samples = 10
        z_dim = 5
        noise = gans.get_noise(n_samples, z_dim, device="cpu")
        self.assertEqual(noise.shape, (n_samples, z_dim))
        self.assertEqual(noise.device, torch.device("cpu"))

    def test_get_noise_cuda(self):
        if torch.cuda.is_available():
            n_samples = 5
            z_dim = 3
            noise = gans.get_noise(n_samples, z_dim, device="cuda")
            self.assertEqual(noise.shape, (n_samples, z_dim))
            self.assertEqual(noise.device, torch.device("cuda"))
        else:
            self.skipTest("CUDA is not available.")


class TestGenerator(TestCase):

    def test_generator_output_shape(self):
        z_dim = 10
        im_dim = 28 * 28  # Assuming MNIST-like images
        hidden_dim = 64
        batch_size = 32
        generator = gans.Generator(z_dim, im_dim, hidden_dim)

        noise = torch.randn(batch_size, z_dim)
        generated_images = generator(noise)

        self.assertEqual(generated_images.shape, (batch_size, im_dim))

    def test_generator_block_output_shape(self):
        input_dim = 10
        output_dim = 20
        generator_block = gans.Generator.get_generator_block(input_dim, output_dim)

        input_tensor = torch.randn(32, input_dim)
        output_tensor = generator_block(input_tensor)

        self.assertEqual(output_tensor.shape, (32, output_dim))


class TestDiscriminator(TestCase):
    def test_discriminator_output_shape(self):
        im_dim = 28 * 28  # Assuming MNIST-like images
        hidden_dim = 64
        batch_size = 32
        discriminator = gans.Discriminator(im_dim, hidden_dim)

        input_images = torch.randn(batch_size, im_dim)
        output = discriminator(input_images)

        self.assertEqual(output.shape, (batch_size, 1))

    def test_discriminator_block_output_shape(self):
        input_dim = 20
        output_dim = 10
        discriminator_block = gans.Discriminator.get_discriminator_block(input_dim, output_dim)

        input_tensor = torch.randn(32, input_dim)
        output_tensor = discriminator_block(input_tensor)

        self.assertEqual(output_tensor.shape, (32, output_dim))


class TestTraining(TestCase):
    data_u = [(torch.randn(2, 1, 1, 1), None)] * 2  # Mock data_u DataLoader
    data_c = [(torch.randn(2, 1, 1, 1), None)] * 2  # Mock data_c DataLoader

    training = gans.Training(
        n_epochs=1,
        z_dim=1,
        display_step=10,
        batch_size=32,
        lr=0.0002,
        c_lambda=10,
        crit_repeats=5,
        data_c=data_c,
        data_u=data_u,
        change_img_ref=100,
        out_dir="output_dir",
        device="cpu",
    )

    # Call the train method
    training.train("experiment")
