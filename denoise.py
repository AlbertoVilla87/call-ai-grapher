import logging
import datetime

from call_ai_grapher.vision import Vision
from call_ai_grapher.autoencoder import Training
from config import denoise_config

import torch

torch.manual_seed(0)


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        img_dir = denoise_config.IM_DIR
        out_dir = denoise_config.EXP_DIR
        current_date_time = datetime.datetime.now()
        date_time_str = current_date_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment = f"{denoise_config.EXP_NAME}_{date_time_str}"
        n_epochs = denoise_config.N_EPOCHS
        batch_size = denoise_config.BATCH_SIZE
        lr = denoise_config.LR
        encoded_space_dim = denoise_config.ENCODED_SPACE_DIM
        noise_factor = denoise_config.NOISE_FACTOR
        image_w = denoise_config.IMAGE_W
        image_h = denoise_config.IMAGE_H

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
