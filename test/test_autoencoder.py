import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from call_ai_grapher.autoencoder import Training, Encoder, Decoder, DataLoader


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.loss_fn = torch.nn.MSELoss()
        self.train_loader = MagicMock(spec=DataLoader)
        self.valid_loader = MagicMock(spec=DataLoader)
        self.test_loader = MagicMock(spec=DataLoader)
        self.n_epochs = 10
        self.lr = 0.001
        self.encoded_space_dim = 100
        self.noise_factor = 0.3
        self.out_dir = "/path/to/output"
        self.train_instance = Training(
            train_loader=self.train_loader,
            val_loader=self.valid_loader,
            test_loader=self.test_loader,
            n_epochs=self.n_epochs,
            lr=self.lr,
            encoded_space_dim=self.encoded_space_dim,
            noise_factor=self.noise_factor,
            out_dir=self.out_dir,
            device=self.device,
        )

    @patch("call_ai_grapher.autoencoder.SummaryWriter")
    @patch("call_ai_grapher.autoencoder.Encoder")
    @patch("call_ai_grapher.autoencoder.Decoder")
    @patch("call_ai_grapher.autoencoder.Training.train_epoch_den")
    @patch("call_ai_grapher.autoencoder.Training.test_epoch_den")
    @patch("call_ai_grapher.autoencoder.Vision.plot_ae_outputs_den")
    def test_train(self, mock_plot, mock_test_epoch, mock_train_epoch, mock_decoder, mock_encoder, mock_writer):
        # Mocks
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        mock_encoder_instance = MagicMock(spec=Encoder)
        mock_encoder.return_value = mock_encoder_instance
        mock_decoder_instance = MagicMock(spec=Decoder)
        mock_decoder.return_value = mock_decoder_instance
        mock_train_epoch.return_value = 0.5
        mock_test_epoch.return_value = 0.6

        # Call the method
        self.train_instance.train("experiment")

        # Assertions
        mock_encoder_instance.to.assert_called_once_with(self.device)
        mock_decoder_instance.to.assert_called_once_with(self.device)
        self.assertEqual(mock_writer_instance.add_scalars.call_count, self.n_epochs)
        self.assertEqual(mock_plot.call_count, self.n_epochs)

    def test_train_epoch_den(self):

        encoder = Encoder(encoded_space_dim=4)
        decoder = Decoder(encoded_space_dim=4)
        params_to_optimize = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]
        optim = torch.optim.Adam(params_to_optimize, lr=self.lr, weight_decay=1e-05)
        dataloader = [(torch.randn(1, 1, 28, 28), None)] * 1

        # Call the method
        loss = Training.train_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device="cpu",
            dataloader=dataloader,
            loss_fn=Training.CRITERION,
            optimizer=optim,
            noise_factor=self.noise_factor,
        )

        # Assertions
        self.assertIsInstance(loss, np.float32)

    def test_test_epoch_den(self):

        encoder = Encoder(encoded_space_dim=4)
        decoder = Decoder(encoded_space_dim=4)
        dataloader = [(torch.randn(1, 1, 28, 28), None)] * 2

        # Call the function
        val_loss = Training.test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device="cpu",
            dataloader=dataloader,
            loss_fn=Training.CRITERION,
            noise_factor=0.3,
        )

        # Ensure the output is a float
        self.assertIsInstance(val_loss, torch.Tensor)

    def test_get_im_dim(self):
        # Create a simulated tensor
        tensor_shape = (10, 3, 64, 64)  # (batch_size, channels, height, width)
        tensor = torch.randn(*tensor_shape)

        # Simulate the data
        data = [(tensor, None)]

        # Call the function
        im_dim = Training.get_im_dim(data)

        # Ensure the result is as expected
        expected_im_dim = tensor_shape[2] * tensor_shape[3]
        self.assertEqual(im_dim, expected_im_dim)
