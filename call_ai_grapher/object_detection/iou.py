class IOU:

    @staticmethod
    def compute_iou(box_candidate: list, box_actual: list, epsilon: float = 1e-5) -> float:
        """
        Intersection Over Union (IOU) measures how overlapping the predicted and
        actual bounding boxes are, while Union measures the overall space possible for
        overlap. IoU is the ratio of the overlapping region between the two bounding
        boxes over the combined region of both the bounding boxes
        :param box_candidate: coordinates of candidate

        Args:
            box_candidate (list): coordinates of candidate
            box_actual (list): coordinates of actual
            epsilon (float, optional): address division by zero. Defaults to 1e-5.

        Returns:
            float: intersection over union
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
