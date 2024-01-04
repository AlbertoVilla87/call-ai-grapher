import os
import re
import glob
import imageio
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Vision:
    @staticmethod
    def show_tensor_images(image_tensor, num_images=1, size=(1, 30, 180)):
        """
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in a uniform grid.
        """
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=1)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    @staticmethod
    def save_image(image_tensor, real_tensor, out_img, num_images=1, size=(1, 30, 180)):
        """_summary_

        :param image_tensor: _description_
        :type image_tensor: _type_
        :param real_tensor: _description_
        :type real_tensor: _type_
        :param out_img: _description_
        :type out_img: _type_
        :param num_images: _description_, defaults to 1
        :type num_images: int, optional
        :param size: _description_, defaults to (1, 28, 28)
        :type size: tuple, optional
        """
        image_unflat = image_tensor.detach().cpu().view(-1, *size)
        real_unflat = real_tensor.detach().cpu().view(-1, *size)
        fake = image_unflat[:num_images]
        fake_bin = (fake >= 0.5).float()
        appended_tensor = torch.cat(
            (
                fake,
                fake_bin,
                real_unflat[:num_images],
            ),
            dim=0,
        )
        image_grid = make_grid(appended_tensor, nrow=1).permute(1, 2, 0).squeeze().numpy()
        plt.imsave(out_img, image_grid, cmap="gray")

    @staticmethod
    def load_images(dir: str, batch_size: int, image_w: int, image_h: int):
        """_summary_
        :param dir: _description_
        :type dir: str
        :param batch_size: _description_
        :type batch_size: int
        :param image_w: _description_
        :type image_w: int
        :param image_h: _description_
        :type image_h: int
        :return: _description_
        :rtype: _type_
        """
        transform = transforms.Compose(
            [
                transforms.Resize((image_h, image_w)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        dataset = DataLoader(datasets.ImageFolder(root=dir, transform=transform), batch_size=batch_size)
        return dataset

    @staticmethod
    def create_gifs(dir_images: str, git_path: str):
        """_summary_
        :param dir_images: _description_
        :type dir_images: str
        :param git_path: _description_
        :type git_path: str
        """
        files = glob.glob(os.path.join(dir_images, "*.png"))
        files = sorted(files, key=Vision.extract_number)
        chunk = 5
        files = [files[i] for i in range(0, len(files), chunk)]
        images = [imageio.imread(file) for file in files]
        imageio.mimsave(git_path, images, duration=0.0005, loop=True)

    @staticmethod
    def extract_number(s):
        s = os.path.basename(s)
        return int(re.search(r"\d+", s).group())

    @staticmethod
    def plot_ae_outputs_den(
        dataset: DataLoader,
        encoder,
        decoder,
        device: str,
        out_img_path: str,
        n_ims_display: int = 10,
        noise_factor: float = 0.3,
    ):
        """_summary_
        :param dataset: _description_
        :type dataset: DataLoader
        :param encoder: _description_
        :type encoder: _type_
        :param decoder: _description_
        :type decoder: _type_
        :param device: _description_
        :type device: str
        :param out_img_path: _description_
        :type out_img_path: str
        :param n_ims_display: _description_, defaults to 10
        :type n_ims_display: int, optional
        :param noise_factor: _description_, defaults to 0.3
        :type noise_factor: float, optional
        """
        plt.figure(figsize=(16, 4.5))
        i = 0
        for img, _ in dataset:
            ax = plt.subplot(3, n_ims_display, i + 1)
            image_noisy = Vision.add_noise(img, noise_factor)
            image_noisy = image_noisy.to(device)

            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                rec_img = decoder(encoder(image_noisy))

            plt.imshow(img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_ims_display // 2:
                ax.set_title("Original images")
            ax = plt.subplot(3, n_ims_display, i + 1 + n_ims_display)
            plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_ims_display // 2:
                ax.set_title("Corrupted images")

            ax = plt.subplot(3, n_ims_display, i + 1 + n_ims_display + n_ims_display)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n_ims_display // 2:
                ax.set_title("Reconstructed images")
            i += 1
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.7, top=0.9, wspace=0.3, hspace=0.3)
        plt.savefig(out_img_path)

    @staticmethod
    def add_noise(inputs: torch, noise_factor: float = 0.3):
        """
        Add noise to the images
        :param inputs: images
        :type inputs: torch
        :param noise_factor: _description_, defaults to 0.3
        :type noise_factor: float, optional
        :return: _description_
        :rtype: _type_
        """
        noisy = inputs + torch.randn_like(inputs) * noise_factor
        noisy = torch.clip(noisy, 0.0, 1.0)
        return noisy
