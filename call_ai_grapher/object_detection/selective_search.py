import cv2
import selectivesearch

from torch_snippets import *
from skimage.segmentation import felzenszwalb


class SelectiveSearch:
    @staticmethod
    def segment_image(image_path: str):
        """
        Extract segments based on the color, texture, size, and shape
        compatibility of content within an image.
        :param image_path: _description_
        :type image_path: str
        """
        img = read(image_path)
        segments = felzenszwalb(img, scale=200)
        return img, segments

    @staticmethod
    def show_segments_image(image_path: str):
        """
        Show segments of the iamge
        :param image_path: _description_
        :type image_path: str
        """
        img, segments = SelectiveSearch.segment_image(image_path)
        print(segments)
        subplots([img, segments], titles=["Original Image", "Segments"], sz=10, nc=2)

    @staticmethod
    def extract_region_proposals(image_path: str, perc: float):
        """
        Identify islands of regions where the pixels are similar to one another.
        :param image_path: _description_
        :type image_path: str
        :param perc: _description_
        :type perc: float
        :return: _description_
        :rtype: _type_
        """
        img = read(image_path)
        _, regions = selectivesearch.selective_search(img)
        img_area = np.prod(img.shape[:2])
        candidates = []
        for r in regions:
            if r["rect"] in candidates:
                continue
            if r["size"] > (perc * img_area):
                continue
            if r["size"] > (1 * img_area):
                continue
            x, y, w, h = r["rect"]
            candidates.append(list(r["rect"]))
        return img, candidates

    @staticmethod
    def show_region_proposals(image_path: str, perc: float):
        """
        Show region proposals
        :param image_path: _description_
        :type image_path: str
        :param area_perc: _description_, defaults to 0.05
        :type area_perc: float, optional
        """
        img, candidates = SelectiveSearch.extract_region_proposals(image_path, perc)
        show(img, bbs=candidates)

    @staticmethod
    def segment_characters_mser(image_path: str, save_image: bool = False):
        """Detect characters using Maximally stable extremal region extractor (MSER)
        :param image_path: _description_
        :type image_path: str
        :param save_image: _description_, defaults to False
        :type save_image: bool, optional
        :return: _description_
        :rtype: _type_
        """
        img = cv2.imread(image_path)
        mser = cv2.MSER_create()

        # Resize the image so that MSER can work better
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detectRegions(gray)
        for region in regions[0]:
            x, y, w, h = cv2.boundingRect(region)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        base_path, extension = image_path.split(".")
        out_path = f"{base_path}_regions.{extension}"
        cv2.namedWindow("img", 0)
        if save_image:
            cv2.imwrite(out_path, vis)
        return regions
