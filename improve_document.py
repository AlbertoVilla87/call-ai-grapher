import argparse
import logging

from call_ai_grapher import CallAIgraher


def _build_detector(args):
    if args.detector == "yolo":
        from call_ai_grapher.pipeline.yolo_detector import YoloCharacterDetector

        return YoloCharacterDetector(args.model, confidence=args.confidence)
    from call_ai_grapher.pipeline.detector import CharacterDetector

    return CharacterDetector()


def _parse_args():
    parser = argparse.ArgumentParser(description="Improve the handwriting of a scanned document")
    parser.add_argument("--input", required=True, help="path to the scanned page image")
    parser.add_argument("--output", required=True, help="path where the improved page is saved")
    parser.add_argument("--alpha", type=float, default=1.0, help="improvement amount in [0, 1]")
    parser.add_argument("--detector", choices=["mser", "yolo"], default="mser", help="character detection backend")
    parser.add_argument("--model", default="models/character_detector.pt", help="YOLO weights path (--detector yolo)")
    parser.add_argument("--confidence", type=float, default=0.25, help="minimum detection confidence (--detector yolo)")
    return parser.parse_args()


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args()
        grapher = CallAIgraher(detector=_build_detector(args))
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
