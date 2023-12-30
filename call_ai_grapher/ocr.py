import logging
import easyocr
import numpy as np

TEXT = "text"
TOP_LEFT = "top_left"
BOTTOM_RIGHT = "bottom_right"
WIDTH = "width"
HEIGHT = "height"


class OCR:
    def __init__(self, lang: str = "en", thres: int = 0.2):
        """_summary_
        :param lang: _description_, defaults to "en"
        :type lang: str, optional
        """
        self.reader = easyocr.Reader([lang, lang])
        self.thres = thres

    def info_text_from_image(self, img_path: str) -> dict:
        """_summary_
        :param img_path: _description_
        :type img_path: str
        :return: _description_
        :rtype: dict
        """
        text = ""
        coordinates = None
        result = self.reader.readtext(img_path)
        for res in result:
            if res[2] > self.thres:
                coordinates_set = res[0]

                if coordinates is None:
                    coordinates = np.array(coordinates_set)
                else:
                    coordinates = np.vstack((coordinates, np.array(coordinates_set)))

                text = text + " " + res[1]

        if text == "":
            logging.warning("Sorry, text not recognized")
            return {}

        top_left = coordinates.min(axis=0)
        bottom_right = coordinates.max(axis=0)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        ocr_info = {TEXT: text, TOP_LEFT: top_left, BOTTOM_RIGHT: bottom_right, WIDTH: width, HEIGHT: height}
        return ocr_info

    @staticmethod
    def get_text(ocr_info: list):
        """_summary_
        :param ocr_info: _description_
        :type ocr_info: list
        """
        return ocr_info[TEXT]

    @staticmethod
    def get_width(ocr_info: list):
        """_summary_
        :param ocr_info: _description_
        :type ocr_info: list
        """
        return ocr_info[WIDTH]

    @staticmethod
    def get_height(ocr_info: list):
        """_summary_
        :param ocr_info: _description_
        :type ocr_info: list
        """
        return ocr_info[HEIGHT]
