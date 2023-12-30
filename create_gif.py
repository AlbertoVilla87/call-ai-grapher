import os
import logging

from call_ai_grapher.vision import Vision


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        img_dir = "fakes/experiment_3"
        out_path = "gif/exp_3.gif"

        Vision.create_gifs(img_dir, out_path)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
