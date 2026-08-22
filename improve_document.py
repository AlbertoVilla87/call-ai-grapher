import argparse
import logging

from call_ai_grapher import CallAIgraher


def _build_detector(args):
    if args.detector == "yolo":
        from call_ai_grapher.pipeline.yolo_detector import YoloCharacterDetector

        return YoloCharacterDetector(args.model, confidence=args.confidence)
    from call_ai_grapher.pipeline.detector import CharacterDetector

    return CharacterDetector()


def _build_classifier(args):
    if not args.classifier:
        return None
    from call_ai_grapher.pipeline.classifier import CharClassifier

    return CharClassifier(args.classifier)


def _build_stylizer(args):
    if args.stylizer == "neural":
        from call_ai_grapher.pipeline.stylizer import NeuralStylizer

        return NeuralStylizer(args.stylizer_model)
    if args.stylizer == "latent":
        from call_ai_grapher.pipeline.stylizer import LatentStylizer

        if not args.classifier:
            raise ValueError("--stylizer latent requires --classifier so each character finds its reference glyph")
        return LatentStylizer(args.ae_model, alphabet_dir=args.alphabet_dir)
    from call_ai_grapher.pipeline.stylizer import Stylizer

    return Stylizer()


def _parse_args():
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
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()
        grapher = CallAIgraher(
            detector=_build_detector(args),
            classifier=_build_classifier(args),
            stylizer=_build_stylizer(args),
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
    _main()
