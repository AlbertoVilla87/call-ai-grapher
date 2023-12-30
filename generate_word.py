import logging

from call_ai_grapher.scrivener import Scrivener


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        path_font = "believe-it-font/BelieveIt-DvLE.ttf"
        font_size = 40

        Scrivener.generate_text("hello word", path_font, font_size)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
