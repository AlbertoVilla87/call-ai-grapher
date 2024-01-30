import logging

from call_ai_grapher.object_detection.selective_search import SelectiveSearch


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        image_path = "documents/experiment_7/test.jpeg"
        segment_detector = SelectiveSearch()
        segment_detector.segment_characters_mser(image_path, save_image=True)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
