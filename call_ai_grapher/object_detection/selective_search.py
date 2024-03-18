import cv2
import selectivesearch

from torch_snippets import *
from skimage.segmentation import felzenszwalb

from object_detection.data_preparation import OpenImages


class SelectiveSearch:
    """_summary_
    Region Proposals Extraction
    Attributes:
    :param fpaths: store files path
    :type fpaths: list
    :param gtbbs: ground truth bounding boxes
    :type gtbbs: list
    :param clss: classes of objects
    :type clss: list
    :param deltas: delta offset of a bounding box with region proposals
    :type deltas: list
    :param rois: region proposal locations
    :type rois: list
    :param ious: IoU of region proposals with ground truths
    :type ious: list
    """

    def __init__(self):
        self.fpaths = []
        self.gtbbs = []
        self.clss = []
        self.deltas = []
        self.rois = []
        self.ious = []

    def fetching_pr_to_gt(self, ds: OpenImages):
        """Fetching region proposals and the ground truth of offset
        :param ds: Ground truth info
        :type ds: OpenImages
        """
        for im, bbs, labels, _ in enumerate(ds):
            H, W, _ = im.shape
            candidates = SelectiveSearch.segment_characters_mser(im)
            candidates = np.array([(x, y, x + w, y + h) for x, y, w, h in candidates])
            ious, rois, clss, deltas = [], [], [], []
            ious = np.array(
                [[SelectiveSearch.compute_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]
            ).T
            for jx, candidate in enumerate(candidates):
                cx, cy, cX, cY = candidate
                candidate_ious = ious[jx]
                best_iou_at = np.argmax(candidate_ious)
                best_iou = candidate_ious[best_iou_at]
                _ = _x, _y, _X, _Y = bbs[best_iou_at]
                if best_iou > 0.3:
                    clss.append(labels[best_iou_at])
                else:
                    clss.append("background")
                delta = np.array([_x - cx, _y - cy, _X - cX, _Y - cY]) / np.array([W, H, W, H])
                deltas.append(delta)
                rois.append(candidate / np.array([W, H, W, H]))

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

    @staticmethod
    def compute_iou(box_candidate: list, box_actual: list, epsilon: float = 1e-5) -> float:
        """Intersection Over Union (IOU) measures how overlapping the predicted and
        actual bounding boxes are, while Union measures the overall space possible for
        overlap. IoU is the ratio of the overlapping region between the two bounding
        boxes over the combined region of both the bounding boxes
        :param box_candidate: coordinates of candidate
        :type box_candidate: list
        :param box_actual: coordinates of actual
        :type box_actual: list
        :param epsilon: address division by zero
        :type epsilon: float
        :return: _description_
        :rtype: float
        """
        x1 = max(box_candidate[0], box_actual[0])
        y1 = max(box_candidate[1], box_actual[1])
        x2 = min(box_candidate[2], box_actual[2])
        y2 = min(box_candidate[3], box_actual[3])
        width = x2 - x1
        height = y2 - y1
        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height
        area_a = (box_candidate[2] - box_candidate[0]) * (box_candidate[3] - box_candidate[1])
        area_b = (box_actual[2] - box_actual[0]) * (box_actual[3] - box_actual[1])
        area_combined = area_a + area_b - area_overlap
        iou = area_overlap / (area_combined + epsilon)
        return iou
