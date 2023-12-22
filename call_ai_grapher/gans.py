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
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    """
    Generator Class
    """

    def __init__(self, z_dim: int = 10, im_chan: int = 1, hidden_dim: int = 64):
        """
        :param z_dim: dimension of the noise vector
        :type z_dim: int, optional
        :param im_chan: the number of channels in the images, fitted for the dataset used
        :type im_chan: int, optional
        :param hidden_dim: the inner dimension, defaults to 128
        :type hidden_dim: int, optional
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_chan = im_chan
        self.hidden_dim = hidden_dim
        self.gen = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """_summary_
        :return: _description_
        :rtype: nn.Sequential
        """
        return nn.Sequential(
            self.make_gen_block(self.z_dim, self.hidden_dim * 4),
            self.make_gen_block(self.hidden_dim * 4, self.hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(self.hidden_dim * 2, self.hidden_dim),
            self.make_gen_block(self.hidden_dim, self.im_chan, kernel_size=4, final_layer=True),
        )

    def forward(self, noise: torch):
        """Given a noise tensor, returns generated images
        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :type noise: torch
        :return: _description_
        :rtype: _type_
        """
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

    def unsqueeze_noise(self, noise: torch):
        """Completing a forward pass of the generate: given a noise tensor,
        returns a copy of that noise with width and heigh = 1 and channel = z_dim
        :param noise: _description_
        :type noise: torch
        :return: _description_
        :rtype: _type_
        """
        return noise.view(len(noise), self.z_dim, 1, 1)

    def make_gen_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        final_layer: bool = False,
    ) -> torch.nn:
        """Function for returning a block of the DCGAN
        :param input_channels: how many channels the input feature representation has
        :type input_channels: int
        :param output_channels: how many channels the output feature representation should have
        :type output_channels: int
        :param kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        :type kernel_size: int
        :param stride: the stride of the convolution
        :type stride: int
        :param final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm), defaults to False
        :type final_layer: bool, optional
        :return: _description_
        :rtype: torch.nn
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:  # Final Layer
            return nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride), nn.Tanh())


class Discriminator(nn.Module):
    """Discrimator Class
    :param nn: _description_
    :type nn: _type_
    """

    def __init__(self, im_chan: int = 1, hidden_dim: int = 16):
        """_summary_
        :param im_chan: the number of channels in the images, fitted for the dataset used, defaults to 784
        :type im_chan: int, optional
        :param hidden_dim: _description_, defaults to 128
        :type hidden_dim: int, optional
        """
        super(Discriminator, self).__init__()
        self.im_chan = im_chan
        self.hidden_dim = hidden_dim
        self.disc = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """_summary_
        :return: _description_
        :rtype: nn.Sequential
        """
        disc = nn.Sequential(
            self.make_gen_block(self.im_chan, self.hidden_dim),
            self.make_gen_block(self.hidden_dim, self.hidden_dim * 2),
            self.make_gen_block(self.hidden_dim * 2, 1, final_layer=True),
        )
        return disc

    def forward(self, image: torch):
        """Given an image tensor, returns a 1-dimension tensor representing fake/real
        :param noise: a noise tensor with dimensions (n_samples, z_dim)
        :type image: image tensor
        :return: _description_
        :rtype: _type_
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

    @property
    def get_disc(self):
        return self.disc

    def make_gen_block(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        final_layer: bool = False,
    ) -> torch.nn:
        """Function for returning a block of the DCGAN
        :param input_channels: how many channels the input feature representation has
        :type input_channels: int
        :param output_channels: how many channels the output feature representation should have
        :type output_channels: int
        :param kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        :type kernel_size: int
        :param stride: the stride of the convolution
        :type stride: int
        :param final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm), defaults to False
        :type final_layer: bool, optional
        :return: _description_
        :rtype: torch.nn
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )


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
        beta_1: float,
        beta_2: float,
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
        :param beta_1: parameter control the optimizer's momentum
        :type beta_1: float
        :param beta_2: parameter control the optimizer's momentum
        :type beta_2: float
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.device = device
        self.data_c = data_c
        self.data_u = data_u
        self.out_dir = out_dir

    def train(self):
        """_summary_"""
        gen = Generator(self.z_dim).to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        disc = Discriminator().to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        gen = gen.apply(Training.weights_init)
        disc = disc.apply(Training.weights_init)
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        data = self.data_u
        for _ in tqdm(range(self.n_epochs)):
            # Dataloader returns the batches
            for real, _ in data:
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.to(self.device)

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

            if cur_step == 2000:
                data = self.data_c
                disc = Discriminator().to(self.device)
                disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr / 100, betas=(self.beta_1, self.beta_2))

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
        gen_op = gen.forward(noise)
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

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
