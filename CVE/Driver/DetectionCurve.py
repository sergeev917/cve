__all__ = (
    'DetectionCurveDriver',
)

from ..Base import IEvaluationDriver
from numpy import (
    empty,
    argsort,
    float32,
    isfinite,
    diff,
    isclose,
    nonzero,
    logical_not,
    concatenate,
    trapz,
)

# NOTE: percent conversion (multiply by 100) is not applied in precision and
#       recall handlers because we're calculating AUC (AP) using those values

# NOTE: entity name is here to help additional plugins like graph plotter to
#       represent the given information: like axis names, percentage/log-scale
#       conversions and so on. Those options are required for usability, but
#       are not ours to set: for example, plotter could use a different locale.

class PrecisionHandler:
    __slots__ = ('curr_detections', 'curr_true_positives')
    entity = 'precision'
    def __init__(self, verifier, verifier_out_lst):
        self.curr_detections = 0
        self.curr_true_positives = 0
    def point(self, interval):
        # detection count delta is the interval elements count
        # TP count delta is the sum over interval (since 0 = FP, 1 = TP)
        self.curr_detections += interval.shape[0]
        self.curr_true_positives += interval.sum()
        # calculating: TP / (TP + FP), (TP + FP) is the detections count
        return self.curr_true_positives / self.curr_detections

class RecallHandler:
    __slots__ = ('inv_gt_size', 'curr_true_positives')
    entity = 'recall'
    def __init__(self, verifier, verifier_out_lst):
        # calculating gt objects count (since it is constant)
        full_gt_count = 0
        for output in verifier_out_lst:
            # gt objects are in TP and FN-count
            delta = output[0].shape[0] + output[2]
            full_gt_count += delta
        self.inv_gt_size = 1. / full_gt_count
        self.curr_true_positives = 0
    def point(self, interval):
        # interval contains payload points per detections
        # so, (TP + FP) is incremented per interval-element
        # and payload indicates which detection is TP
        self.curr_true_positives += interval.sum()
        # calculating: TP / (TP + FN), (TP + FN) is the gt set size
        return self.curr_true_positives * self.inv_gt_size

class FPCountHandler:
    __slots__ = ('curr_false_positives')
    entity = 'false-positives'
    def __init__(self, verifier, verifier_out_lst):
        self.curr_false_positives = 0
    def point(self, interval):
        # false positives count is the count of 0 in interval (0/1 are allowed)
        self.curr_false_positives += (interval.shape[0] - interval.sum())
        return self.curr_false_positives

class FPPIHandler:
    __slots__ = ('curr_false_positives', 'inv_samples_count')
    entity = 'false-positives-per-sample'
    def __init__(self, verifier, verifier_out_lst):
        # calculating gt samples count (since it is constant)
        # NOTE: it is different (rather than gt objects) because one sample
        #       (which is corresponding to one image, for example) could have
        #       multiple gt objects (bboxes, for instance)
        self.inv_samples_count = 1. / len(verifier_out_lst)
        self.curr_false_positives = 0
    def point(self, interval):
        # false positives count is the count of 0 in interval (0/1 are allowed)
        self.curr_false_positives += (interval.shape[0] - interval.sum())
        # calculating: FP / (samples count)
        return self.curr_false_positives * self.inv_samples_count

available_handlers = {
    'precision': PrecisionHandler,
    'recall': RecallHandler,
    'fp': FPCountHandler,
    'fppi': FPPIHandler,
}

class DetectionCurveDriver(IEvaluationDriver):
    __slots__ = ('axis_classes', 'verifier', 'enable_auc')
    def __init__(self, **kwargs):
        axis_drv = kwargs.get('mode', ('precision', 'recall'))
        if axis_drv[0] not in available_handlers:
            raise Exception('unknown notion "{}" is requested for v-axis'.format(axis_drv[0])) # FIXME
        if axis_drv[1] not in available_handlers:
            raise Exception('unknown notion "{}" is requested for h-axis'.format(axis_drv[1])) # FIXME
        self.axis_classes = tuple(map(available_handlers.__getitem__, axis_drv))
        self.enable_auc = axis_drv == ('precision', 'recall')
    def reconfigure(self, verifier_instance):
        if verifier_instance.interpret_as != 'confidence-relative':
            raise Exception('only confidence-relative input is supported') # FIXME
        self.verifier = verifier_instance
    def collect(self, verifier_output):
        # gathering size of confidence array to create:
        # we will use FP and TP confidence points
        need_conf_size = lambda out: out[0].shape[0] + out[1].shape[0]
        conf_size = sum(map(need_conf_size, verifier_output))
        if conf_size == 1:
            # no actual points to be processed
            raise Exception('no actual data to process') # FIXME
        # creating X/Y axis value handlers for the given verifier responses
        init_drv = lambda cls: cls(self.verifier, verifier_output)
        axis_drv = tuple(map(init_drv, self.axis_classes))
        # preallocating array for confidences and payloads of the required size
        # NOTE: payload is simply `is_true_positive` indicator (for now)
        # NOTE: no numpy.concatenate, it will require a sequence (list/tuple)
        confs = empty(conf_size, order = 'C', dtype = 'float32')
        payload = empty(conf_size, order = 'C', dtype = 'int32')
        curr_idx = 0
        # pushing verification results (per-sample) into one shared array
        for sample_result in verifier_output:
            # processing confidences of true positives
            tp_confs = sample_result[0]
            tp_count = tp_confs.shape[0]
            if tp_count > 0:
                next_idx = curr_idx + tp_count
                confs[curr_idx : next_idx] = tp_confs
                payload[curr_idx : next_idx].fill(1)
                curr_idx = next_idx
            # processing confidences of false positives
            fp_confs = sample_result[1]
            fp_count = fp_confs.shape[0]
            if fp_count > 0:
                next_idx = curr_idx + fp_count
                confs[curr_idx : next_idx] = fp_confs
                payload[curr_idx : next_idx].fill(0)
                curr_idx = next_idx
        # checking for values which we will not tolerate
        if not isfinite(confs).all():
            raise Exception('Inf/NaN values as a confidence are prohibited')
        # sorting by confidence to do thesholding on it later (descending)
        # FIXME: check mergesort instead of quicksort
        sort_indices = argsort(confs, kind = 'quicksort')[::-1]
        confs = confs[sort_indices]
        payload = payload[sort_indices]
        del sort_indices
        # thresholding confidence level from the highest one
        # NOTE: indices on the following line are corresponding to the points
        #       where there is a gap compare to the next value
        target_indices = nonzero(logical_not(isclose(diff(confs), 0)))[0] # FIXME: configurable
        # adding fictive first point to produce correct range and the last point
        # NOTE: +1 is to use non-inclusive index of a range end
        target_indices = concatenate(([-1], target_indices, [conf_size])) + 1
        points_count = len(target_indices) - 1
        x_points = empty(points_count, dtype = 'float32')
        y_points = empty(points_count, dtype = 'float32')
        for idx in range(1, len(target_indices)):
            interval = payload[target_indices[idx - 1] : target_indices[idx]]
            point_idx = idx - 1
            y_points[point_idx] = axis_drv[0].point(interval)
            x_points[point_idx] = axis_drv[1].point(interval)
        results = {
            'x-points': x_points,
            'y-points': y_points,
            'x-entity': self.axis_classes[1].entity,
            'y-entity': self.axis_classes[0].entity,
        }
        if self.enable_auc:
            fair_auc = trapz(y_points, x_points)
            # since the data could start from an initial recall value (the first
            # group of the highest-confidence detections), we are missing the
            # most left area. We will fill the gap between the initial recall
            # and zero recall with the constant precision which is corresponding
            # to the its initial value.
            filled_auc = y_points[0] * x_points[0]
            results['auc'] = fair_auc + filled_auc
            results['filled_auc_delta'] = filled_auc
        return results
