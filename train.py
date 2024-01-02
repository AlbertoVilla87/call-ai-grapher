import logging
import datetime

from call_ai_grapher.vision import Vision
from call_ai_grapher.gans import Training

import torch

torch.manual_seed(0)


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        corr_dir = "scrivener_words_GloriousFree-dBR6"
        unc_dir = "scrivener_words_ArianaVioleta-dz2K"
        out_dir = "fakes/experiment_4"
        current_date_time = datetime.datetime.now()
        date_time_str = current_date_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment = f"exp_4_{date_time_str}"
        n_epochs = 2000
        z_dim = 256
        image_w = 100
        image_h = 40
        display_step = 10
        batch_size = 2
        lr = 0.00001
        change_img_ref = 1000
        data_correct = Vision.load_images(corr_dir, batch_size, image_w, image_h)
        data_uncorrect = Vision.load_images(unc_dir, batch_size, image_w, image_h)

        trainer = Training(
            n_epochs, z_dim, display_step, batch_size, lr, data_correct, data_uncorrect, change_img_ref, out_dir
        )
        trainer.train(experiment)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
