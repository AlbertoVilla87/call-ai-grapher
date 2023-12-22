import logging

from call_ai_grapher.vision import Vision
from call_ai_grapher.gans import Training

import torch

torch.manual_seed(0)


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        corr_dir = "handwriting"
        unc_dir = "myhandw"
        out_dir = "fakes/experiment_2"
        n_epochs = 5000
        z_dim = 64
        display_step = 10
        batch_size = 2
        lr = 0.005
        beta_1 = 0.5
        beta_2 = 0.999
        data_correct = Vision.load_images(corr_dir, batch_size=2)
        data_uncorrect = Vision.load_images(unc_dir, batch_size=2)

        trainer = Training(
            n_epochs, z_dim, display_step, batch_size, lr, beta_1, beta_2, data_correct, data_uncorrect, out_dir
        )
        trainer.train()

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
