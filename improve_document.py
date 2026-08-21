import argparse
import logging

from call_ai_grapher import CallAIgraher


def _parse_args():
    parser = argparse.ArgumentParser(description="Improve the handwriting of a scanned document")
    parser.add_argument("--input", required=True, help="path to the scanned page image")
    parser.add_argument("--output", required=True, help="path where the improved page is saved")
    parser.add_argument("--alpha", type=float, default=1.0, help="improvement amount in [0, 1]")
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()
        grapher = CallAIgraher()
        result = grapher.improve_document(args.input, args.output, args.alpha)
        logging.info(
            "Done: %d characters processed, output at %s",
            len(result.chars),
            result.output_path,
        )
    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
