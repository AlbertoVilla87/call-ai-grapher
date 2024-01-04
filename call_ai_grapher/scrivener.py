import os
from PIL import Image, ImageDraw, ImageFont

from call_ai_grapher import ocr
from call_ai_grapher.ocr import OCR


class Scrivener:
    @staticmethod
    def generate_text(text: str, font_style: str, font_size):
        """_summary_
        :param text: _description_
        :type text: str
        :param font_style: _description_
        :type font_style: str
        :param size_image: _description_
        :type size_image: int
        """
        reader = OCR()
        size_image = (1, 1)
        image = Image.new("RGB", size_image, color="white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_style, font_size)
        _, _, w, h = draw.textbbox((0, 0), text, font=font)
        image = Image.new("RGB", (w, font_size), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, fill="black", font=font)
        out_path = Scrivener.create_path(text, font_style)
        image.save(out_path)
        text_info = reader.info_text_from_image(out_path)
        if text_info:
            delta_w = text_info[ocr.BOTTOM_RIGHT][0] - w
            image.crop((0, 0, w - delta_w, h)).save(out_path)
        return out_path

    @staticmethod
    def create_path(text: str, font_style: str) -> str:
        """_summary_
        :param text: _description_
        :type text: str
        :param font_style: _description_
        :type font_style: str
        :return: _description_
        :rtype: str
        """
        style = os.path.splitext(os.path.basename(font_style))[0]
        if not os.path.exists(f"scrivener_words_{style}"):
            os.makedirs(f"scrivener_words_{style}")
            os.makedirs(f"scrivener_words_{style}/images")
        path = f"scrivener_words_{style}/images/{text}.png"
        return path
