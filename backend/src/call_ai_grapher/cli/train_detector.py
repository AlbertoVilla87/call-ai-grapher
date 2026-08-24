import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the YOLO character detector")
    parser.add_argument("--data", default="dataset/detector/data.yaml", help="dataset data.yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="base model or pretrained weights")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--imgsz", type=int, default=480, help="training image size")
    parser.add_argument("--out", default="models/character_detector.pt", help="where to copy the best weights")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
        args = _parse_args(argv)

        model = YOLO(args.model)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, verbose=True)

        best = Path(model.trainer.best)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, out_path)
        logging.info("Best weights copied to %s", out_path)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    main()
