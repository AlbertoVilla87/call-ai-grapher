import cv2
import json
import numpy as np
import pandas as pd

IMAGE_ID = "id"
CATEGORY_ID = "id"
CATEGORY_BOX = "category_id"
IMAGE_CAT_ID = "image_id"
CATEGORY_NAME = "name"
IMAGES = "images"
BOX = "bbox"
ANNOTATIONS = "annotations"
FILE_NAME = "file_name"
LABEL_NAME = "label_name"
CATEGORIES = "categories"
X_MIN = "x_min"
Y_MIN = "y_min"
X_MAX = "x_max"
Y_MAX = "y_max"

# TODO: refactor OpenImages to adapt Coco format


def read_json(path: str) -> dict:
    with open(path, "r") as file:
        data = json.load(file)
    return data


class GroundTruthData:
    def __init__(self, data_path: str):
        """_summary_
        :param data_path: _description_
        :type data_path: str
        :return: _description_
        :rtype: _type_
        """
        self.data = read_json(data_path)

    def get_name_category(self, cat_id: int) -> str:
        """
        Get the name of a category box
        Args:
            cat_id (int): id number

        Returns:
            str: name of label
        """
        cats = self.data[CATEGORIES]
        for cat in cats:
            if cat[CATEGORY_ID] == cat_id:
                return cat[CATEGORY_NAME]

    def get_coords_bbx_image_id(self, image_id: int) -> list:
        """
        Get the coordinates boxes of an specific image
        Args:
            image_id (int): _description_

        Returns:
            list: list of coordinates
        """
        boxes = []
        anns = self.data[ANNOTATIONS]
        for ann in anns:
            if ann[IMAGE_CAT_ID] == image_id:
                coord_x0 = ann[BOX][0]
                coord_x1 = ann[BOX][0] + ann[BOX][2]
                coord_y0 = ann[BOX][1]
                coord_y1 = ann[BOX][1] + ann[BOX][3]
                boxes.append([coord_x0, coord_y0, coord_x1, coord_y1])
        return boxes

    def get_cat_bbx_image_id(self, image_id: int) -> list:
        """
        Get the category boxes of an specific image
        Args:
            image_id (int): _description_

        Returns:
            list: list of coordinates
        """
        cats = []
        anns = self.data[ANNOTATIONS]
        for ann in anns:
            if ann[IMAGE_CAT_ID] == image_id:
                cats.append(ann[CATEGORY_BOX])
        return cats

    def get_images_id(self) -> list:
        """
        Get all the id of the labeled images

        Returns:
            list: _description_
        """
        images = self.data[IMAGES]
        ids = []
        for image in images:
            ids.append(image[IMAGE_ID])
        return ids

    def get_images_file(self) -> list:
        """
        Get all the filenames of the labeled images

        Returns:
            list: _description_
        """
        images = self.data[IMAGES]
        names = []
        for image in images:
            names.append(image[FILE_NAME])
        return names


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
        self.images_id = self.get_images_id()
        self.images_name = self.get_images_file()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, ix):
        image_name = self.images_name[ix]
        image_id = self.images_id[ix]
        image_path = f"{self.root}/{image_name}"
        image = cv2.imread(image_path, 1)[..., ::-1]  # convert BGR to RGB
        boxes = self.get_coords_bbx_image_id(image_id)
        classes = self.get_cat_bbx_image_id(image_id)
        return image, boxes, classes, image_path
