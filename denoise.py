import logging
import datetime

from call_ai_grapher.vision import Vision
from call_ai_grapher.autoencoder import Training

import torch

torch.manual_seed(0)


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        img_dir = "scrivener_words_GloriousFree-dBR6"
        out_dir = "denoise/experiment_6"
        current_date_time = datetime.datetime.now()
        date_time_str = current_date_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment = f"exp_6_{date_time_str}"
        n_epochs = 200
        batch_size = 1
        lr = 0.001
        encoded_space_dim = 4
        noise_factor = 0.3
        image_w = 28
        image_h = 28

        train_dataset = Vision.load_images(img_dir, batch_size, image_w, image_h)
        val_dataset = Vision.load_images(img_dir, batch_size, image_w, image_h)
        test_dataset = Vision.load_images(img_dir, batch_size, image_w, image_h)

        trainer = Training(
            train_dataset,
            val_dataset,
            test_dataset,
            n_epochs,
            lr,
            encoded_space_dim,
            noise_factor,
            out_dir,
        )
        trainer.train(experiment)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
