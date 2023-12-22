import os
import numpy as np
import re
import glob
import imageio
import logging
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Vision:
    @staticmethod
    def show_tensor_images(image_tensor, num_images=1, size=(1, 28, 28)):
        """
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in a uniform grid.
        """
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()

    @staticmethod
    def save_image(image_tensor, real_tensor, out_img, num_images=1):
        """_summary_

        :param image_tensor: _description_
        :type image_tensor: _type_
        :param real_tensor: _description_
        :type real_tensor: _type_
        :param out_img: _description_
        :type out_img: _type_
        :param num_images: _description_, defaults to 1
        :type num_images: int, optional
        :param size: _description_,
        :type size: tuple, optional
        """
        try:
            resize = transforms.Compose(
                [
                    transforms.Resize((real_tensor.shape[2], real_tensor.shape[3])),
                ]
            )
            image_tensor = resize(image_tensor)
            image_unflat = image_tensor.detach().cpu()
            real_unflat = real_tensor.detach().cpu()
            appended_tensor = torch.cat(
                (
                    image_unflat[:num_images],
                    real_unflat[:num_images],
                ),
                dim=0,
            )
            image_grid = make_grid(appended_tensor, nrow=1).permute(1, 2, 0).squeeze().numpy()
            min_val = np.min(image_grid)
            max_val = np.max(image_grid)
            normalized_image = (image_grid - min_val) / (max_val - min_val)
            plt.imsave(out_img, normalized_image, cmap="gray")
        except Exception as e:
            logging.info(e)
            pass

    @staticmethod
    def load_images(dir: str, batch_size: int):
        """_summary_
        :param dir: _description_
        :type dir: str
        :param batch_size: _description_
        :type batch_size: int
        :return: _description_
        :rtype: _type_
        """
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
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

    def extract_number(s):
        s = os.path.basename(s)
        return int(re.search(r"\d+", s).group())
