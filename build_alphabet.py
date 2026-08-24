import argparse
import logging

from call_ai_grapher.dataset import (
    DEFAULT_ALPHABET,
    AlphabetDatasetBuilder,
    class_counts,
    find_fonts,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build the Spanish alphabet dataset")
    parser.add_argument("--fonts", default="fonts/**/*.ttf", help="glob pattern with the pretty style .ttf fonts")
    parser.add_argument("--samples", default=None, help="optional directory with real handwriting crops per class")
    parser.add_argument("--out", default="dataset/alphabet", help="output dataset directory")
    parser.add_argument("--size", type=int, default=64, help="normalized image size in pixels")
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()

        builder = AlphabetDatasetBuilder(chars=DEFAULT_ALPHABET, size=args.size)

        fonts = find_fonts(args.fonts)
        if fonts:
            counts = builder.build_from_fonts(fonts, args.out)
            logging.info("Rendered alphabet from %d fonts", len(fonts))
        else:
            logging.warning("No fonts found matching %s; drop your pretty style fonts there", args.fonts)

        if args.samples:
            builder.ingest_samples(args.samples, args.out)
            logging.info("Ingested real handwriting samples from %s", args.samples)

        total = sum(class_counts(args.out).values())
        logging.info("Dataset ready at %s with %d images across %d classes", args.out, total, len(DEFAULT_ALPHABET))

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
