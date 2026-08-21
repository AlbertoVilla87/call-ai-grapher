import argparse
import logging

from call_ai_grapher.dataset.page_generator import write_detector_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a YOLO-format character detection dataset")
    parser.add_argument("--glyphs", default="dataset/alphabet", help="normalized alphabet dataset directory")
    parser.add_argument("--out", default="dataset/detector", help="output dataset directory")
    parser.add_argument("--train-pages", type=int, default=200, help="number of training pages")
    parser.add_argument("--val-pages", type=int, default=40, help="number of validation pages")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()
        write_detector_dataset(
            glyph_dir=args.glyphs,
            out_dir=args.out,
            train_pages=args.train_pages,
            val_pages=args.val_pages,
            seed=args.seed,
        )
    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
