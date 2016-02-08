__all__ = ('BoundingBoxIoUVerifier',)

from .BoundingBoxCmp import find_best_iou
from ..Base import IVerifier
from ..Annotation.Capability import (
    bounding_box_capability,
    confidence_capability,
    blacklist_capability,
)
from numpy import (
    finfo,
    float32,
    empty,
    argmax,
    append,
    apply_along_axis,
    flatnonzero,
    logical_and,
)

class BoundingBoxIoUVerifier(IVerifier):
    __slots__ = (
        'get_base_bbox',
        'get_test_bbox',
        'get_non_blacklisted',
        'store_info_prepare',
        'threshold',
        'interpret_as'
    )

    def reconfigure(self, gt_dataset, eval_dataset):
        gt_storage_class = gt_dataset.storage_class
        eval_storage_class = eval_dataset.storage_class
        self.get_base_bbox = bounding_box_capability(gt_storage_class)
        self.get_test_bbox = bounding_box_capability(eval_storage_class)
        try:
            get_non_blacklisted = blacklist_capability(gt_storage_class)
        except: # FIXME
            # no blacklist is given, so everything is not blacklisted
            def get_non_blacklisted(sample_markup):
                elem_count = self.get_base_bbox(sample_markup).shape[0]
                non_blacklisted = ndarray((elem_count,),
                                          dtype = 'bool',
                                          shape = 'C')
                non_blacklisted.fill(True)
                return non_blacklisted
        self.get_non_blacklisted = get_non_blacklisted
        try:
            # if confidence output is available then we will return
            # confidences at which TP/FP are produced.
            get_conf = confidence_capability(eval_storage_class)
            def store_info_prepare(test_sample):
                confids = get_conf(test_sample)
                def store_info(indices):
                    return apply_along_axis(confids.__getitem__, 0, indices)
                return store_info
            self.interpret_as = 'confidence-relative'
        except: # FIXME
            # if there is no confidence output we will simply return
            # count of FP and TP instead of their confidences.
            self.interpret_as = 'unconditional-numbers'
            def store_info_prepare(test_sample):
                def store_info(indices):
                    return len(indices)
                return store_info
        self.store_info_prepare = store_info_prepare

    def __init__(self, threshold = None):
        self.threshold = threshold
        if self.threshold == None:
            self.threshold = finfo(float32).eps

    def __call__(self, base_sample, test_sample):
        base_set = self.get_base_bbox(base_sample)
        test_set = self.get_test_bbox(test_sample)
        non_blacklisted_gt = self.get_non_blacklisted(base_sample) # FIXME
        indices, scores = find_best_iou(base_set, test_set)
        # resetting indices of low-iou matches to '-1' (so they become FP)
        indices[scores < self.threshold] = -1
        # false positives are accounted by checking no-gt-mapped detections
        fp_idx = flatnonzero(indices == -1)
        gt_size = base_set.shape[1]
        # NOTE: tp_idx uses preallocated storage to prevent reallocations:
        #       maximum count of tp is gt_size, usage is tracked by tp_idx_top
        tp_idx = empty(gt_size, order = 'C', dtype = 'int32')
        tp_idx_top = fn_count = multicount = 0
        # checking for TP/FN records: only non-blacklisted records are processed
        # FIXME: could be optimized
        for idx in flatnonzero(non_blacklisted_gt):
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
