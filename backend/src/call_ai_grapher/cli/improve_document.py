import argparse
import logging

from call_ai_grapher import CallAIgraher
from call_ai_grapher.pipeline.factory import (
    build_classifier,
    build_detector,
    build_stylizer,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Improve the handwriting of a scanned document")
    parser.add_argument("--input", required=True, help="path to the scanned page image")
    parser.add_argument("--output", required=True, help="path where the improved page is saved")
    parser.add_argument("--alpha", type=float, default=1.0, help="improvement amount in [0, 1]")
    parser.add_argument("--detector", choices=["mser", "yolo"], default="mser", help="character detection backend")
    parser.add_argument("--model", default="models/character_detector.pt", help="YOLO weights path (--detector yolo)")
    parser.add_argument("--confidence", type=float, default=0.25, help="minimum detection confidence (--detector yolo)")
    parser.add_argument("--classifier", default=None, help="classifier checkpoint to label characters (optional)")
    parser.add_argument(
        "--stylizer", choices=["baseline", "neural", "latent"], default="baseline", help="stylization backend"
    )
    parser.add_argument(
        "--stylizer-model", default="models/char_stylizer.pt", help="stylizer checkpoint (--stylizer neural)"
    )
    parser.add_argument(
        "--ae-model", default="models/char_autoencoder.pt", help="autoencoder checkpoint (--stylizer latent)"
    )
    parser.add_argument(
        "--alphabet-dir", default="dataset/alphabet", help="alphabet dataset with the pretty reference glyphs"
    )
    return parser.parse_args(argv)


def main(argv=None):
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args(argv)
        grapher = CallAIgraher(
            detector=build_detector(args.detector, args.model, args.confidence),
            classifier=build_classifier(args.classifier),
            stylizer=build_stylizer(
                args.stylizer,
                stylizer_model=args.stylizer_model,
                autoencoder_model=args.ae_model,
                alphabet_dir=args.alphabet_dir,
                classifier_model=args.classifier,
            ),
        )
        result = grapher.improve_document(args.input, args.output, args.alpha)
        logging.info(
            "Done: %d characters processed, output at %s",
            len(result.chars),
            result.output_path,
        )
    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    main()
