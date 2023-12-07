import os
import logging

from call_ai_grapher.vision import Vision
from call_ai_grapher.gans import Training


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        corr_dir = "handwriting"
        unc_dir = "myhandw"
        out_dir = "fakes"
        n_epochs = 5000
        z_dim = 64
        display_step = 10
        batch_size = 2
        lr = 0.00001
        data_correct = Vision.load_images(corr_dir, batch_size=2)
        data_uncorrect = Vision.load_images(unc_dir, batch_size=2)

        trainer = Training(n_epochs, z_dim, display_step, batch_size, lr, data_correct, data_uncorrect, out_dir)
        trainer.train()

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
