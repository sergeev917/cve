__all__ = ('BBoxIoU',)

from ..Annotation.Capability import bounding_box_capability
from numpy import (
    minimum,
    maximum,
    empty,
    prod,
    argmax,
)

class BBoxIoU:
    __slots__ = ('get_bbox', 'threshold')

    def __init__(self, AnnotationClass, threshold):
        self.get_bbox = bounding_box_capability(AnnotationClass)
        self.threshold = threshold

    def __call__(self, base_sample, test_sample):
        # converting samples to bounding boxes in numpy format
        base_sample = self.get_bbox(base_sample)
        test_sample = self.get_bbox(test_sample)
        # computing areas of bounding boxes in base_sample
        base_areas = prod(base_sample[2:4, :] - base_sample[0:2, :], axis = 0)
        # for every bounding box in test sample compute iou scores
        bboxes_to_test = test_sample.shape[1]
        indices = empty(bboxes_to_test, order = 'C', dtype = 'int32')
        scores = empty(bboxes_to_test, order = 'C', dtype = 'float32')
        zeros = empty((2, 1), order = 'C', dtype = 'int32')
        zeros.fill(0)
        for bbox_idx in range(0, bboxes_to_test):
            # calculating max(0, min(ends) - max(starts))
            tested_bbox = test_sample[:, bbox_idx : bbox_idx + 1]
            diff = minimum(base_sample[2:4,:], tested_bbox[2:4,:]) - \
                   maximum(base_sample[0:2,:], tested_bbox[0:2,:])
            intersection_areas = prod(maximum(zeros, diff), axis = 0)
            tested_bbox_area = prod(tested_bbox[2:4] - tested_bbox[0:2])
            union_areas = tested_bbox_area + base_areas - intersection_areas
            iou_scores = intersection_areas / union_areas
            max_iou_idx = argmax(iou_scores)
            max_iou_score = iou_scores[max_iou_idx]
            if iou_scores[max_iou_idx] < self.threshold:
                max_iou_idx = -1
            indices[bbox_idx] = max_iou_idx
            scores[bbox_idx] = max_iou_score
        return (indices, scores)
