__all__ = ('find_best_iou',)

from numpy import (
    minimum,
    maximum,
    empty,
    prod,
    argmax,
)

#@profile
def find_best_iou(base_set, test_set):
    # NOTE: float32 type will be used in BoundingBoxVerifier
    # NOTE: output index will point to the bbox with maximum iou score
    #       even when it is zero.
    # computing areas of bounding boxes in base_set
    base_areas = prod(base_set[2:4, :] - base_set[0:2, :], axis = 0)
    # for every bounding box in test sample compute iou scores
    bboxes_to_test = test_set.shape[1]
    indices = empty(bboxes_to_test, order = 'C', dtype = 'int32')
    scores = empty(bboxes_to_test, order = 'C', dtype = 'float32')
    zeros = empty((2, 1), order = 'C', dtype = 'int32')
    zeros.fill(0)
    # FIXME: try numpy.vectorize here
    for bbox_idx in range(0, bboxes_to_test):
        # calculating max(0, min(ends) - max(starts))
        tested_bbox = test_set[:, bbox_idx : bbox_idx + 1]
        diff = minimum(base_set[2:4,:], tested_bbox[2:4,:]) - \
               maximum(base_set[0:2,:], tested_bbox[0:2,:])
        intersection_areas = prod(maximum(zeros, diff), axis = 0)
        tested_bbox_area = prod(tested_bbox[2:4] - tested_bbox[0:2])
        union_areas = tested_bbox_area + base_areas - intersection_areas
        iou_scores = intersection_areas / union_areas
        max_iou_idx = argmax(iou_scores)
        max_iou_score = iou_scores[max_iou_idx]
        indices[bbox_idx] = max_iou_idx
        scores[bbox_idx] = max_iou_score
    return (indices, scores)
