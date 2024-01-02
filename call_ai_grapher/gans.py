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
    """
    Generator Class
    """

    def __init__(self, z_dim: int, im_dim: int, hidden_dim: int):
        """
        :param z_dim: _description_,
        :type z_dim: int, optional
        :param im_dim: _description_,
        :type im_dim: int, optional
        :param hidden_dim: _description_,
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

    def __init__(self, im_dim: int, hidden_dim: int):
        """_summary_
        :param im_dim: _description_, defaults to 5400
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


class WGANGP:
    @staticmethod
    def get_gradient(crit: Discriminator, real: torch, fake: torch, epsilon: float) -> float:
        """Return the gradient of the discriminator's scores with respect to mixes of real and fake images
        :param crit: the discriminator model
        :type crit: Discriminator
        :param real: a batch of real images
        :type real: torch
        :param fake: a batch of real fakes
        :type fake: torch
        :param epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        :type epsilon: float
        :return: the gradient of the critic's scores, with respect to the mixed image
        :rtype: float
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
        """Return the gradient penalty, given a gradient. Given a batch of image gradients,
        you calculate the magnitude of eadh image's gradient and penalize the mean quadratic
        distance of each magnitude to 1
        :param gradient: the gradient of the discriminator's scores, with respect to the
        mixed image
        :type gradient: float
        :return: the gradient penalty
        :rtype: float
        """
        # Flatten the gradients so that each row captures one image
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
        """_summary_
        :param n_epochs: _description_
        :type n_epochs: int
        :param z_dim: _description_
        :type z_dim: int
        :param display_step: _description_
        :type display_step: int
        :param batch_size: _description_
        :type batch_size: int
        :param lr: _description_
        :type lr: float
        :param c_lambda: _description_
        :type c_lambda: int
        :param crit_repeats: _description_
        :type crit_repeats: int
        :param data_c: _description_
        :type data_c: DataLoader
        :param data_u: _description_
        :type data_u: DataLoader
        :param change_img_ref: _description_
        :type change_img_ref: int
        :param out_dir: _description_
        :type out_dir: str
        :param device: _description_, defaults to "cpu"
        :type device: str, optional
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
        :param experiment: _description_
        :type experiment: str
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
        gen_loss = -torch.mean(disc_op_fake)
        return gen_loss

    @staticmethod
    def get_disc_loss(
        gen: Generator, disc: Discriminator, c_lambda: int, real: list, num_images: int, z_dim: int, device: str
    ):
        """Return the loss of the discriminator given inputs.
        :param gen: _description_
        :type gen: Generator
        :param disc: _description_
        :type disc: Discriminator
        :param c_lambda: _description_
        :type c_lambda: int
        :param real: _description_
        :type real: list
        :param num_images: _description_
        :type num_images: int
        :param z_dim: _description_
        :type z_dim: int
        :param device: _description_
        :type device: str
        :return: _description_
        :rtype: _type_
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
