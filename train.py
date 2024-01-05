import logging
import datetime

from call_ai_grapher.vision import Vision
from call_ai_grapher.gans import Training
from config import gans_config

import torch

torch.manual_seed(0)


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        corr_dir = gans_config.CORR_DIR
        unc_dir = gans_config.UNDIR
        out_dir = gans_config.EXP_DIR
        current_date_time = datetime.datetime.now()
        date_time_str = current_date_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment = f"{gans_config.EXP_NAME}_{date_time_str}"
        n_epochs = gans_config.N_EPOCHS
        z_dim = gans_config.Z_DIM
        image_w = gans_config.IMAGE_W
        image_h = gans_config.IMAGE_H
        display_step = gans_config.DISPLAY_STEP
        batch_size = gans_config.BATCH_SIZE
        lr = gans_config.LR
        change_img_ref = gans_config.CHANGE_IMG_REF
        c_lambda = gans_config.C_LAMBDA
        crit_repeats = gans_config.CRIT_REPEATS
        data_correct = Vision.load_images(corr_dir, batch_size, image_w, image_h)
        data_uncorrect = Vision.load_images(unc_dir, batch_size, image_w, image_h)

        trainer = Training(
            n_epochs,
            z_dim,
            display_step,
            batch_size,
            lr,
            c_lambda,
            crit_repeats,
            data_correct,
            data_uncorrect,
            change_img_ref,
            out_dir,
        )
        trainer.train(experiment)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
