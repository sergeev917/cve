__all__ = ('BoundingBoxIoUVerifier',)

from .BoundingBoxCmp import find_best_iou
from .Verifier import IVerifier
from ..Annotation.Capability import (
    bounding_box_capability,
    confidence_capability,
)
from numpy import (
    finfo,
    float32,
    empty,
    argmax,
    append,
    apply_along_axis,
    flatnonzero,
)

# FIXME: use separate annotation-classes and get_conf/get_bbox
class BoundingBoxIoUVerifier(IVerifier):
    __slots__ = ('get_bbox', 'store_info_prepare', 'threshold', 'interpret_as')

    def __init__(self, AnnotationClass, threshold = None):
        self.get_bbox = bounding_box_capability(AnnotationClass)
        try:
            # if confidence output is available then we will return
            # confidences at which TP/FP are produced.
            self.interpret_as = 'confidence-relative'
            get_conf = confidence_capability(AnnotationClass)
            def store_info_prepare(test_sample):
                confids = get_conf(test_sample)
                def store_info(indices):
                    return apply_along_axis(confids.__getitem__, 0, indices)
                return store_info
        except: # FIXME
            # if there is no confidence output we will simply return
            # count of FP and TP instead of their confidences.
            self.interpret_as = 'unconditional-numbers'
            def store_info_prepare(test_sample):
                def store_info(indices):
                    return len(indices)
                return store_info
        self.store_info_prepare = store_info_prepare
        self.threshold = threshold
        if self.threshold == None:
            self.threshold = finfo(float32).eps
        # FIXME: ignore flag usage!

    def __call__(self, base_sample, test_sample):
        base_set = self.get_bbox(base_sample)
        test_set = self.get_bbox(test_sample)
        indices, scores = find_best_iou(base_set, test_set)
        # resetting low-iou matches indices to '-1'
        indices[scores < self.threshold] = -1
        # FIXME: filtering ignore-marked bboxes
        # mapping matches to ground-truth info (which is base_sample)
        gt_size = base_set.shape[1]
        fp_idx = flatnonzero(indices == -1)
        # NOTE: tp_idx uses preallocated storage to prevent reallocations:
        #       maximum count of tp is gt_size, usage is tracked by tp_idx_top
        tp_idx = empty(gt_size, order = 'C', dtype = 'int32')
        tp_idx_top = 0
        fn_count = 0
        multicount = 0
        for idx in range(0, gt_size):
            match_indices = indices == idx
            if match_indices.any():
                match_iou_scores = scores[match_indices]
                multicount += (match_iou_scores.shape[0] - 1)
                selected_idx = argmax(match_iou_scores)
                tp_idx[tp_idx_top] = selected_idx
                tp_idx_top += 1
            else:
                fn_count += 1
        store_info = self.store_info_prepare(test_sample)
        return (
            store_info(tp_idx[:tp_idx_top]),
            store_info(fp_idx),
            fn_count,
            multicount,
        )
