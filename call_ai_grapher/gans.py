import torch
import logging
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from call_ai_grapher.vision import Vision


def get_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.randn:
    """Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution
    :param n_samples: the number of samples to generate
    :type n_samples: int
    :param z_dim: the dimension of the noise vector
    :type z_dim: int
    :param device: _description_, defaults to "cpu"
    :type device: str, optional
    :return: _description_
    :rtype: torch.randn
    """
    return torch.randn(n_samples, z_dim).to(device)


class Generator(nn.Module):
    """
    Generator Class
    """

    def __init__(self, z_dim: int = 10, im_dim: int = 784, hidden_dim: int = 128):
        """
        :param z_dim: _description_, defaults to 10
        :type z_dim: int, optional
        :param im_dim: _description_, defaults to 784
        :type im_dim: int, optional
        :param hidden_dim: _description_, defaults to 128
        :type hidden_dim: int, optional
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim
        self.gen = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """_summary_
        :return: _description_
        :rtype: nn.Sequential
        """
        gen = nn.Sequential(
            Generator.get_generator_block(self.z_dim, self.hidden_dim),
            Generator.get_generator_block(self.hidden_dim, self.hidden_dim * 2),
            Generator.get_generator_block(self.hidden_dim * 2, self.hidden_dim * 4),
            Generator.get_generator_block(self.hidden_dim * 4, self.hidden_dim * 8),
            nn.Linear(self.hidden_dim * 8, self.im_dim),
            nn.Sigmoid(),
        )
        return gen

    def forward(self, noise: torch):
        """Given a noise tensor, returns generated images
        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :type noise: torch
        :return: _description_
        :rtype: _type_
        """
        return self.gen(noise)

    @property
    def get_gen(self):
        return self.gen

    @staticmethod
    def get_generator_block(input_dim: int, output_dim: int) -> torch.nn:
        """Function for returning a block of the generator's neural network
           given input and output dimensions
        :param input_dim: _description_
        :type input_dim: int
        :param output_dim: _description_
        :type output_dim: int
        :return: _description_
        :rtype: torch.nn
        """
        network = nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(inplace=True))
        return network


class Discriminator(nn.Module):
    """Discrimator Class
    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self, im_dim: int = 784, hidden_dim: int = 128):
        """_summary_
        :param im_dim: _description_, defaults to 784
        :type im_dim: int, optional
        :param hidden_dim: _description_, defaults to 128
        :type hidden_dim: int, optional
        """
        super(Discriminator, self).__init__()
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim
        self.disc = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """_summary_
        :return: _description_
        :rtype: nn.Sequential
        """
        disc = nn.Sequential(
            Discriminator.get_discriminator_block(self.im_dim, self.hidden_dim * 4),
            Discriminator.get_discriminator_block(self.hidden_dim * 4, self.hidden_dim * 2),
            Discriminator.get_discriminator_block(self.hidden_dim * 2, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        return disc

    def forward(self, image: torch):
        """Given an image tensor, returns a 1-dimension tensor representing fake/real
        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :type image: image tensor
        :return: _description_
        :rtype: _type_
        """
        return self.disc(image)

    @property
    def get_disc(self):
        return self.disc

    @staticmethod
    def get_discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
        """Discriminator Block
        :param input_dim: the dimension of the input vector
        :type input_dim: int
        :param output: the dimension of the output vector
        :type output: int
        :return: a discriminator neural network layer, with a linear transformation
        :rtype: nn.Sequential
        """
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2))


class Training:
    # Loss function
    CRITERION = nn.BCEWithLogitsLoss()

    def __init__(
        self,
        n_epochs: int,
        z_dim: int,
        display_step: int,
        batch_size: int,
        lr: float,
        data_c: DataLoader,
        data_u: DataLoader,
        out_dir: str,
        device: str = "cpu",
    ):
        """_summary_
        :param n_epochs: the number of times you iterate through the entire dataset when training
        :type n_epochs: int
        :param z_dim: the dimension of the noise vector
        :type z_dim: int
        :param display_step: how often to display/visualize the images
        :type display_step: int
        :param batch_size: the number of images per forward/backward pass
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param data: tensors images
        :type data: DataLoader
        :param out_dir: fakes images generated directory
        :type out_dir: str
        :param device: device type
        :type device: str
        """
        self.n_epochs = n_epochs
        self.z_dim = z_dim
        self.display_step = display_step
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.data_c = data_c
        self.data_u = data_u
        self.out_dir = out_dir

    def train(self):
        """_summary_"""
        gen = Generator(self.z_dim).to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr)
        disc = Discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr)
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        data = self.data_u
        for _ in tqdm(range(self.n_epochs)):
            # Dataloader returns the batches
            for real, _ in data:
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(self.device)

                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()

                # Calculate discriminator loss
                disc_loss = Training.get_disc_loss(gen, disc, real, cur_batch_size, self.z_dim, self.device)

                # Update gradients
                disc_loss.backward(retain_graph=True)

                # Update optimizer
                disc_opt.step()

                gen_opt.zero_grad()
                gen_loss = Training.get_gen_loss(gen, disc, self.batch_size, self.z_dim, self.device)
                gen_loss.backward(retain_graph=True)
                gen_opt.step()

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / self.display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.display_step

                ### Visualization code ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    logging.info(
                        f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}"
                    )
                    fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
                    fake = gen(fake_noise)
                    Vision.save_image(fake, real, f"{self.out_dir}/{cur_step}.png")
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1

            if cur_step == 500:
                data = self.data_c
                disc = Discriminator().to(self.device)
                disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr)
                mean_discriminator_loss = 0
                self.lr = self.lr / 10

    @staticmethod
    def get_gen_loss(gen: Generator, disc: Discriminator, num_images: int, z_dim: int, device: str = "cpu") -> float:
        """Return the loss of the generator given inputs
        :param gen: generator model, withc returns an image given z-dimensional noise
        :type gen: Generator
        :param disc: discriminator model, which returns a single-dimensional prediction of real/fake
        :type disc: Discriminator
        :param num_images: the number of images the generated should produce,
        which is also the lenght of the real images
        :type num_images: int
        :param z_dim: the dimension of the noise vector
        :type z_dim: int
        :param device: the device type, defaults to "cpu"
        :type device: str, optional
        :return: _description_
        :rtype: float
        """
        noise = get_noise(num_images, z_dim, device=device)
        gen_op = gen(noise)
        disc_op_fake = disc(gen_op)
        gen_loss = Training.CRITERION(disc_op_fake, torch.ones_like(disc_op_fake))
        return gen_loss

    @staticmethod
    def get_disc_loss(gen: Generator, disc: Discriminator, real: list, num_images: int, z_dim: int, device: str):
        """Return the loss of the discriminator given inputs.
        :param gen: _description_
        :type gen: Generator
        :param disc: _description_
        :type disc: Discriminator
        :param real: _description_
        :type real: list
        :param num_images: _description_
        :type num_images: int
        :param z_dim: _description_
        :type z_dim: int
        :param device: _description_
        :type device: str
        """
        noise = get_noise(num_images, z_dim, device=device)
        gen_op = gen(noise)
        disc_op_fake = disc(gen_op.detach())
        disc_loss_fake = Training.CRITERION(disc_op_fake, torch.zeros_like(disc_op_fake))
        disc_op_real = disc(real)
        disc_loss_real = Training.CRITERION(disc_op_real, torch.ones_like(disc_op_real))
        disc_loss = (disc_loss_fake + disc_loss_real) / 2
        return disc_loss
