import logging

from call_ai_grapher.scrivener import Scrivener


def _main():
    try:
        logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

        out_font = "fonts/glorious-free-font/GloriousFree-dBR6.ttf"
        font_size = 40

        words = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        for word in words:
            Scrivener.generate_text(word, out_font, font_size)

    except Exception:
        logging.exception("Process failed")


if __name__ == "__main__":
    _main()
