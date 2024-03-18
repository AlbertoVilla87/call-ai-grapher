import cv2
import numpy as np
import pandas as pd


IMAGE_ID = "image_id"
LABEL_NAME = "label_name"
X_MIN = "x_min"
Y_MIN = "y_min"
X_MAX = "x_max"
Y_MAX = "y_max"

# TODO: refactor OpenImages to adapt Coco format


class GroundTruthData:
    def __init__(self, data_path: str):
        """_summary_
        :param data_path: _description_
        :type data_path: str
        :return: _description_
        :rtype: _type_
        """
        self.data = pd.read_csv(data_path, sep=";")

    def get_coordinates(self) -> list:
        return self.data[X_MIN, Y_MIN, X_MAX, Y_MAX].values

    def get_unique_labels(self) -> list:
        return self.data[IMAGE_ID].unique()

    def get_labels(self) -> list:
        return self.data[LABEL_NAME].values.tolist()


class OpenImages(GroundTruthData):
    def __init__(self, data_label_path: str, image_folder: str):
        """_summary_
        :param data_label_path: _description_
        :type data_label_path: str
        :param image_folder: _description_
        :type image_folder: str
        """
        super().__init__(data_label_path)
        self.root = image_folder
        self.unique_images = self.get_labels()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, ix):
        image_id = self.unique_images[ix]
        image_path = f"{self.root}/{image_id}.jpg"
        image = cv2.imread(image_path, 1)[..., ::-1]  # conver BGR to RGB
        h, w, _ = image.shape
        df = self.data.copy()
        df = df[df[IMAGE_ID] == image_id]
        boxes = self.get_labels
        boxes = (boxes * np.array([w, h, w, h])).astype(np.uint16).tolist()
        classes = df[LABEL_NAME].values.tolist()
        return image, boxes, classes, image_path
