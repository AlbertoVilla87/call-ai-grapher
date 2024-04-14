import logging
from torch_snippets import *

from call_ai_grapher.object_detection import data_preparation


def _main():
    try:
        logging.basicConfig(
            format="%(asctime)-15s %(levelname)s %(message)s",
            level=logging.INFO,
        )
        ground_path = "data/input/documents/experiment_8/annotations/coco.json"
        image_dir = "data/input/documents/experiment_8/docs"
        ground = data_preparation.OpenImages(ground_path, image_dir)
        im, bbs, clss, _ = ground[0]
        show(im, bbs=bbs, texts=clss, sz=10)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
