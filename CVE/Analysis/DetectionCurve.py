__all__ = (
    'DetectionPerformanceCurve',
    'CurveData2d',
    'MeanAvgPrecisionData',
)

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
from collections import namedtuple
from ..Base import DetectionConfidenceAssessment
from ..FlowBuilder import ResourceTypeInfo
from ..Logger import get_default_logger

CurveData2d = namedtuple(
    'CurveData2d',
    ['x_points', 'x_entity', 'y_points', 'y_entity'],
)

MeanAvgPrecisionData = namedtuple(
    'MeanAvgPrecisionData',
    ['auc', 'autofilled_auc'],
)

# NOTE: percent conversion (multiply by 100) is not applied in precision and
#       recall handlers because we're calculating AUC (AP) using those values

# NOTE: entity name is here to help additional plugins like graph plotter to
#       represent the given information: like axis names, percentage/log-scale
#       conversions and so on. Those options are required for usability, but
#       are not ours to set: for example, plotter could use a different locale.

# NOTE: handlers use the processing assessments list while initialization; this
#       is because some global constants might be needed to produce the results
#       while accessing some intervals of detections list

class PrecisionHandler:
    entity = 'precision'
    def __init__(self, assessment_list):
        # don't need to do anything with assessment_list
        self.detections_count = 0
        self.tp_count = 0
    def point(self, tp_flags_interval):
        # tp_flags_interval: 0/1 per detection, 1 = tp, 0 = fp
        self.detections_count += tp_flags_interval.shape[0]
        self.tp_count += tp_flags_interval.sum()
        # calculating: TP / (TP + FP), (TP + FP) is the detections count
        return self.tp_count / self.detections_count

class RecallHandler:
    entity = 'recall'
    def __init__(self, assessment_list):
        # calculating gt objects count (since it is constant)
        # assessment_list: (tp_confs, fp_confs, fn_count, multicount)
        get_gt_count = lambda e: e[0].shape[0] + e[2]
        self.inverted_gt_count = 1. / sum(map(get_gt_count, assessment_list))
        self.tp_count = 0
    def point(self, tp_flags_interval):
        self.tp_count += tp_flags_interval.sum()
        # calculating: TP / (TP + FN), (TP + FN) is the gt set size
        return self.tp_count * self.inverted_gt_count

class FPCountHandler:
    entity = 'false-positives'
    def __init__(self, assessment_list):
        self.fp_count = 0
    def point(self, tp_flags_interval):
        # need to calculate number of '0' having '1' as the only other option
        self.fp_count += (tp_flags_interval.shape[0] - tp_flags_interval.sum())
        return self.fp_count

class FPPIHandler:
    entity = 'false-positives-per-sample'
    def __init__(self, assessment_list):
        # calculating samples count (not gt markup count)
        self.inverted_samples_count = 1. / len(assessment_list)
        self.fp_count = 0
    def point(self, tp_flags_interval):
        # false positives count is the count of 0 in interval (0/1 are allowed)
        self.fp_count += (tp_flags_interval.shape[0] - tp_flags_interval.sum())
        # calculating: FP / (samples count)
        return self.fp_count * self.inverted_samples_count

# FIXME: we can use flow builder to avoid multiple calculation of the same data

available_handlers = {
    'precision': PrecisionHandler,
    'recall': RecallHandler,
    'fp': FPCountHandler,
    'fppi': FPPIHandler,
}

class DetectionPerformanceCurve:
    def __init__(self, **kwargs):
        self._logger = kwargs.get('logger', get_default_logger())
        axis_drv = kwargs.get('mode', ('precision', 'recall'))
        if axis_drv[0] not in available_handlers:
            raise Exception('unknown notion "{}" is requested for v-axis'.format(axis_drv[0])) # FIXME
        if axis_drv[1] not in available_handlers:
            raise Exception('unknown notion "{}" is requested for h-axis'.format(axis_drv[1])) # FIXME
        curve_res = 'analysis:object-detection:curve-{}({})'.format(*axis_drv)
        if axis_drv == ('precision', 'recall'):
            self._map_is_possible = True
            output = (curve_res, 'analysis:object-detection:mean-avg-precision')
        else:
            self._map_is_possible = False
            output = (curve_res,)
        self._contract = (('assessment-list:object-detection',), output)
        self._axis_cls = tuple(map(available_handlers.__getitem__, axis_drv))
    def targets_to_inject(self):
        return self._contract[1]
    def static_contracts(self):
        return [self._contract]
    def get_contract(self, mode_id):
        assert mode_id == 0
        return self._contract
    def setup(self, mode_id, input_types, output_mask):
        assert mode_id == 0
        if not any(output_mask):
            return lambda *args: None, (None,) * len(output_mask)
        input_t = input_types[0]
        if input_t.type_cls is not list:
            raise RuntimeError()
        try:
            elem_cls = input_t.aux_info['elem_type'][0].type_cls
        except (KeyError, IndexError):
            raise RuntimeError()
        if elem_cls is not DetectionConfidenceAssessment:
            raise RuntimeError()
        # now we know that we have a list of DetectionConfidenceAssessment,
        # which is the only information representation we support
        axis_classes = self._axis_cls
        map_is_possible = self._map_is_possible
        if map_is_possible:
            map_is_requested = output_mask[1]
            output_types = (
                ResourceTypeInfo(CurveData2d),
                ResourceTypeInfo(MeanAvgPrecisionData),
            )
        else:
            output_types = (ResourceTypeInfo(CurveData2d),)
        def worker(assessment_list):
            # element (DetectionConfidenceAssessment) is a named tuple with
            # the following fields in the order:
            #   tp_confs, fp_confs, fn_count, multicount
            conf_points_count = lambda e: e[0].shape[0] + e[1].shape[0]
            points_count = sum(map(conf_points_count, assessment_list))
            if points_count <= 1:
                raise RuntimeError('not enough data to produce curve')
            # initializing axis processors (need to precalculate constants)
            axis_drv = tuple(map(lambda cl: cl(assessment_list), axis_classes))
            # payload is simply `is_true_positive` indicator (for now)
            # NOTE: no numpy.concatenate, it will require a sequence (list/tuple)
            confidences = empty(points_count, order = 'C', dtype = 'float32')
            payload = empty(points_count, order = 'C', dtype = 'int32')
            arr_top_idx = 0
            for sample_info in assessment_list:
                # processing confidences of true positives
                tp_confs = sample_info[0]
                tp_count = tp_confs.shape[0]
                if tp_count > 0:
                    next_top_idx = arr_top_idx + tp_count
                    confidences[arr_top_idx : next_top_idx] = tp_confs
                    payload[arr_top_idx : next_top_idx].fill(1)
                    arr_top_idx = next_top_idx
                # processing confidences of false positives
                fp_confs = sample_info[1]
                fp_count = fp_confs.shape[0]
                if fp_count > 0:
                    next_top_idx = arr_top_idx + fp_count
                    confidences[arr_top_idx : next_top_idx] = fp_confs
                    payload[arr_top_idx : next_top_idx].fill(0)
                    arr_top_idx = next_top_idx
            # checking for values which we will not tolerate
            if not isfinite(confidences).all():
                raise RuntimeError('Inf/NaN confidences are prohibited')
            # sorting by confidence to perform thesholding (descending)
            # FIXME: check mergesort instead of quicksort
            sort_indices = argsort(confidences, kind = 'quicksort')[::-1]
            confidences = confidences[sort_indices]
            payload = payload[sort_indices]
            del sort_indices
            # thresholding confidence level from the highest one
            # NOTE: indices on the following line are corresponding to the points
            #       where there is a gap compare to the next value
            target_indices = nonzero(logical_not(isclose(diff(confidences), 0)))[0] # FIXME: configurable
            # adding fictive first point to produce correct range and the last point
            # NOTE: +1 is to use non-inclusive index of a range end
            target_indices = concatenate(([-1], target_indices, [points_count])) + 1
            points_count = len(target_indices) - 1
            x_points = empty(points_count, dtype = 'float32')
            y_points = empty(points_count, dtype = 'float32')
            for idx in range(1, len(target_indices)):
                interval = payload[target_indices[idx - 1] : target_indices[idx]]
                point_idx = idx - 1
                y_points[point_idx] = axis_drv[0].point(interval)
                x_points[point_idx] = axis_drv[1].point(interval)
            curve_data = CurveData2d(
                x_points,
                axis_classes[1].entity,
                y_points,
                axis_classes[0].entity,
            )
            if not map_is_possible:
                return curve_data
            if not map_is_requested:
                # we announced mAP calculation capability, but no interest
                return curve_data, None
            # and here, we can calculate mAP and it is requested
            fair_auc = trapz(y_points, x_points)
            # since the data could start from an initial recall value (the first
            # group of the highest-confidence detections), we are missing the
            # most left area. We will fill the gap between the initial recall
            # and zero recall with the constant precision which is corresponding
            # to the its initial value.
            filled_auc = y_points[0] * x_points[0]
            map_data = MeanAvgPrecisionData(fair_auc + filled_auc, filled_auc)
            return curve_data, map_data
        if self._logger.__dummy__:
            return worker, output_types
        # wrap into logger here
        def log_wrapped_worker(*args, **kwargs):
            with self._logger.subtask('analysing data to produce PR-curve'):
                return worker(*args, **kwargs)
        return log_wrapped_worker, output_types
