import logging

from call_ai_grapher.scrivener import Scrivener


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        in_font = "quicksand/Quicksand-Regular.otf"
        out_font = "freedom-font/Freedom-10eM.ttf"
        font_size = 40

        Scrivener.generate_text("alberto", out_font, font_size)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
