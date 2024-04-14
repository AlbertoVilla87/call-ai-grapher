import cv2
import selectivesearch
from object_detection.data_preparation import OpenImages
from skimage.segmentation import felzenszwalb
from torch_snippets import *


class SelectiveSearch:

    def __init__(self):
        """
        Region Proposals Extraction
        """
        # Store files path
        self.fpaths = []
        # Ground truth bounding boxes
        self.gtbbs = []
        # Classes of objects
        self.clss = []
        # Delta offset of a bounding box with region proposals
        self.deltas = []
        # Region proposal locations
        self.rois = []
        # IoU of region proposals with ground truths
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
