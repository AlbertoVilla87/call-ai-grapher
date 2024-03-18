import torch
import logging
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

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

    def __init__(self, z_dim: int, im_dim: int, hidden_dim: int):
        """
        Generator class

        Args:
            z_dim (int): the dimension of the noise vector, a scalar
            im_dim (int): the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
            hidden_dim (int): the inner dimension, a scalar
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim
        self.gen = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """
        Build the neural network

        Returns:
            nn.Sequential: network
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
        """
        Given a noise tensor, returns generated images

        Args:
            noise (torch): _description_

        Returns:
            _type_: _description_
        """
        return self.gen(noise)

    @property
    def get_gen(self):
        return self.gen

    @staticmethod
    def get_generator_block(input_dim: int, output_dim: int) -> torch.nn:
        """
        Function for returning a block of the generator's neural network
        given input and output dimensions

        Args:
            input_dim (int): _description_
            output_dim (int): _description_

        Returns:
            torch.nn: _description_
        """
        network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
        return network


class Discriminator(nn.Module):

    def __init__(self, im_dim: int, hidden_dim: int):
        """
        Discriminator Class

        Args:
            im_dim (int): the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
            hidden_dim (int): the inner dimension, a scalar
        """
        super(Discriminator, self).__init__()
        self.im_dim = im_dim
        self.hidden_dim = hidden_dim
        self.disc = self.create_nn()

    def create_nn(self) -> nn.Sequential:
        """
        Build the neural network

        Returns:
            nn.Sequential: network
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
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.

        Args:
            image (torch): a flattened image tensor with dimension (im_dim)

        Returns:
            _type_: _description_
        """
        return self.disc(image)

    @property
    def get_disc(self):
        return self.disc

    @staticmethod
    def get_discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
        """
        Return a sequence of operations corresponding to a discriminator block of DCGAN

        Args:
            input_dim (int): _description_
            output_dim (int): _description_

        Returns:
            nn.Sequential: _description_
        """
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2))


class WGANGP:
    @staticmethod
    def get_gradient(crit: Discriminator, real: torch, fake: torch, epsilon: float) -> float:
        """
        Return the gradient of the discriminator's scores with respect to mixes of real and fake images

        Args:
            crit (Discriminator): the discriminator model
            real (torch): a batch of real images
            fake (torch): a batch of fake images
            epsilon (float): a vector of the uniformly random proportions of real/fake per mixed image

        Returns:
            float: the gradient of the critic's scores, with respect to the mixed image
        """
        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = crit(mixed_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            # This documentation may be useful, but it should not be necessary:
            # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
            inputs=mixed_images,
            outputs=mixed_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    @staticmethod
    def gradient_penalty(gradient: float) -> float:
        """
        Return the gradient penalty, given a gradient. Given a batch of image gradients,
        you calculate the magnitude of each image's gradient and penalize the mean quadratic
        distance of each magnitude to 1

        Args:
            gradient (float): the gradient of the discriminator's scores, with respect to the
        mixed image

        Returns:
            float: the gradient penalty
        """
        gradient = gradient.view(len(gradient), -1)
        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty


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
        c_lambda: int,
        crit_repeats: int,
        data_c: DataLoader,
        data_u: DataLoader,
        change_img_ref: int,
        out_dir: str,
        device: str = "cpu",
    ):
        """
        Training GANs

        Args:
            n_epochs (int): number of epochs
            z_dim (int): dimension of noise vector
            display_step (int): how often to display/visualize the images
            batch_size (int): the number of images per forward/backward pass
            lr (float): learning rate
            c_lambda (int): _description_
            crit_repeats (int): _description_
            data_c (DataLoader): machine-encoded alphabet
            data_u (DataLoader): handwriting alphabet
            change_img_ref (int): change reference iteration
            out_dir (str): output directory
            device (str, optional): _description_. Defaults to "cpu".
        """
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.n_epochs = n_epochs
        self.z_dim = z_dim
        self.display_step = display_step
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.data_c = data_c
        self.data_u = data_u
        self.out_dir = out_dir
        self.im_dim_xyz = Training.get_im_dim(data_c)
        self.im_dim = self.im_dim_xyz[1] * self.im_dim_xyz[2]
        self.change_img_ref = change_img_ref
        self.c_lambda = c_lambda
        self.crit_repeats = crit_repeats

    def train(self, experiment: str):
        """
        Train GANS

        Args:
            experiment (str): name of experiment
        """
        writer = SummaryWriter(comment="-" + experiment)
        gen = Generator(self.z_dim, self.im_dim, hidden_dim=500).to(self.device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        disc = Discriminator(im_dim=self.im_dim, hidden_dim=500).to(self.device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        cur_step = 0
        data = self.data_u
        generator_losses = []
        critic_losses = []
        for _ in tqdm(range(self.n_epochs)):
            # Dataloader returns the batches
            for real, _ in data:
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(self.device)

                mean_iteration_critic_loss = 0
                for _ in range(self.crit_repeats):
                    disc_opt.zero_grad()
                    disc_loss = Training.get_disc_loss(
                        gen, disc, self.c_lambda, real, cur_batch_size, self.z_dim, self.device
                    )
                    mean_iteration_critic_loss += disc_loss.item() / self.crit_repeats
                    disc_loss.backward(retain_graph=True)
                    disc_opt.step()

                critic_losses += [mean_iteration_critic_loss]

                # update generator
                gen_opt.zero_grad()
                gen_loss = Training.get_gen_loss(gen, disc, self.batch_size, self.z_dim, self.device)
                gen_loss.backward(retain_graph=True)
                gen_opt.step()

                generator_losses += [gen_loss.item()]

                ### Visualization code ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    gen_mean = sum(generator_losses[-self.display_step :]) / self.display_step
                    crit_mean = sum(critic_losses[-self.display_step :]) / self.display_step
                    writer.add_scalars(
                        "LOSS",
                        {
                            "mean_discriminator_loss": crit_mean,
                            "mean_generator_loss": gen_mean,
                        },
                        global_step=cur_step,
                    )
                    logging.info(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {crit_mean}")
                    fake_noise = get_noise(cur_batch_size, self.z_dim, device=self.device)
                    fake = gen(fake_noise)
                    Vision.save_image(fake, real, f"{self.out_dir}/{cur_step}.png", size=self.im_dim_xyz)

                cur_step += 1

            if cur_step == self.change_img_ref:
                data = self.data_c
        writer.close

    @staticmethod
    def get_gen_loss(gen: Generator, disc: Discriminator, num_images: int, z_dim: int, device: str = "cpu") -> float:
        """
        Loss of the generator given inputs

        Args:
            gen (Generator): generator model, with returns an image given z-dimensional noise
            disc (Discriminator): discriminator model, which returns a single-dimensional prediction of real/fake
            num_images (int): the number of images the generated should produce,
            which is also the length of the real images
            z_dim (int): the dimension of the noise vector
            device (str, optional): _description_. Defaults to "cpu".

        Returns:
            float: _description_
        """
        noise = get_noise(num_images, z_dim, device=device)
        gen_op = gen(noise)
        disc_op_fake = disc(gen_op)
        gen_loss = -torch.mean(disc_op_fake)
        return gen_loss

    @staticmethod
    def get_disc_loss(
        gen: Generator, disc: Discriminator, c_lambda: int, real: list, num_images: int, z_dim: int, device: str
    ):
        """
        Return the loss of the discriminator given inputs.

        Args:
            gen (Generator): _description_
            disc (Discriminator): _description_
            c_lambda (int): _description_
            real (list): _description_
            num_images (int): number of images
            z_dim (int): the dimension of the noise vector
            device (str): _description_

        Returns:
            _type_: _description_
        """
        epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
        noise = get_noise(num_images, z_dim, device=device)
        gen_op = gen(noise)
        disc_op_fake = disc(gen_op.detach())
        disc_op_real = disc(real)
        gradient = WGANGP.get_gradient(disc, real, gen_op.detach(), epsilon)
        gp = WGANGP.gradient_penalty(gradient)
        disc_loss = torch.mean(disc_op_fake) - torch.mean(disc_op_real) + gp * c_lambda
        return disc_loss

    def get_im_dim(data):
        for real, _ in data:
            return (real.shape[1], real.shape[2], real.shape[3])
