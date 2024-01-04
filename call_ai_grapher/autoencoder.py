import numpy as np  # this module is useful to work with numerical arrays
import logging
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from torch import nn

from call_ai_grapher.vision import Vision


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim: int):
        """_summary_
        :param encoded_space_dim: _description_
        :type encoded_space_dim: int
        """
        super(Encoder, self).__init__()
        self.encoded_space_dim = encoded_space_dim
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )
        # review 32 for size -> 128/encoded_space_dim
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(nn.Linear(3 * 3 * 32, 128), nn.ReLU(True), nn.Linear(128, encoded_space_dim))

    def forward(self, image: torch):
        """_summary_
        :param image: _description_
        :type image: torch
        :return: _description_
        :rtype: _type_
        """
        out = self.encoder_cnn(image)
        out = self.flatten(out)
        out = self.encoder_lin(out)
        return out


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim: int):
        """_summary_
        :param encoded_space_dim: _description_
        :type encoded_space_dim: int
        """
        super(Decoder, self).__init__()
        self.encoded_space_dim = encoded_space_dim

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128), nn.ReLU(True), nn.Linear(128, 3 * 3 * 32), nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        """_summary_
        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class Training:
    # Loss function
    CRITERION = torch.nn.MSELoss()

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        n_epochs: int,
        lr: float,
        encoded_space_dim: int,
        noise_factor: float,
        out_dir: str,
        device: str = "cpu",
    ):
        """_summary_
        :param train_loader: _description_
        :type train_loader: DataLoader
        :param val_loader: _description_
        :type val_loader: DataLoader
        :param test_loader: _description_
        :type test_loader: DataLoader
        :param n_epochs: _description_
        :type n_epochs: int
        :param lr: _description_
        :type lr: float
        :param encoded_space_dim: _description_
        :type encoded_space_dim: int
        :param noise_factor: _description_
        :type noise_factor: float
        :param device: _description_, defaults to "cpu"
        :type device: str, optional
        """
        self.train_loader = train_loader
        self.valid_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.noise_factor = noise_factor
        self.encoded_space_dim = encoded_space_dim
        self.device = device
        self.im_dim = Training.get_im_dim(train_loader)
        self.out_dir = out_dir

    def train(self, experiment: str):
        """
        Train Autoencoder
        :param experiment: _description_
        :type experiment: str
        """
        writer = SummaryWriter(comment="-" + experiment)
        encoder = Encoder(encoded_space_dim=self.encoded_space_dim)
        decoder = Decoder(encoded_space_dim=self.encoded_space_dim)
        params_to_optimize = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]
        optim = torch.optim.Adam(params_to_optimize, lr=self.lr, weight_decay=1e-05)
        encoder.to(self.device)
        decoder.to(self.device)
        history_da = {"train_loss": [], "val_loss": []}

        for epoch in tqdm(range(self.n_epochs)):
            logging.info("EPOCH %d/%d" % (epoch + 1, self.n_epochs))
            ### Training (use the training function)
            logging.info("Training...")
            train_loss = Training.train_epoch_den(
                encoder=encoder,
                decoder=decoder,
                device=self.device,
                dataloader=self.train_loader,
                loss_fn=Training.CRITERION,
                optimizer=optim,
                noise_factor=self.noise_factor,
            )
            ### Validation (use the testing function)
            logging.info("Validation...")
            val_loss = Training.test_epoch_den(
                encoder=encoder,
                decoder=decoder,
                device=self.device,
                dataloader=self.valid_loader,
                loss_fn=Training.CRITERION,
                noise_factor=self.noise_factor,
            )
            # Print Validationloss
            history_da["train_loss"].append(train_loss)
            history_da["val_loss"].append(val_loss)
            logging.info(
                "\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}".format(
                    epoch + 1, self.n_epochs, train_loss, val_loss
                )
            )
            writer.add_scalars(
                "LOSS",
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                global_step=epoch,
            )
            out_path = f"{self.out_dir}/epoch_{epoch+1}.png"
            Vision.plot_ae_outputs_den(
                self.test_dataset,
                encoder,
                decoder,
                self.device,
                out_path,
                n_ims_display=10,
                noise_factor=self.noise_factor,
            )

    @staticmethod
    def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
        # Set train mode for both the encoder and the decoder
        encoder.train()
        decoder.train()
        train_loss = []
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for (
            image_batch,
            _,
        ) in tqdm(
            dataloader
        ):  # with "_" we just ignore the labels (the second element of the dataloader tuple)
            # Move tensor to the proper device
            image_noisy = Vision.add_noise(image_batch, noise_factor)
            image_batch = image_batch.to(device)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Evaluate loss
            loss = loss_fn(decoded_data, image_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, noise_factor=0.3):
        # Set evaluation mode for encoder and decoder
        encoder.eval()
        decoder.eval()
        with torch.no_grad():  # No need to track the gradients
            # Define the lists to store the outputs for each batch
            conc_out = []
            conc_label = []
            for image_batch, _ in tqdm(dataloader):
                # Move tensor to the proper device
                image_noisy = Vision.add_noise(image_batch, noise_factor)
                image_noisy = image_noisy.to(device)
                # Encode data
                encoded_data = encoder(image_noisy)
                # Decode data
                decoded_data = decoder(encoded_data)
                # Append the network output and the original image to the lists
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            # Create a single tensor with all the values in the lists
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
        return val_loss.data

    @staticmethod
    def get_im_dim(data):
        for real, _ in data:
            return real.shape[2] * real.shape[3]
