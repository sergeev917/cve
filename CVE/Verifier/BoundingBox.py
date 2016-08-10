__all__ = ('BoundingBoxIoUVerifier',)

from .BoundingBoxCmp import find_best_iou
from ..Annotation.Capability import (
    bounding_box_capability,
    confidence_capability,
    whitelist_capability,
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
from operator import itemgetter
from ..FlowBuilder import ResourceTypeInfo
from ..Base import DetectionSimpleAssessment, DetectionConfidenceAssessment

class BoundingBoxIoUVerifier:
    def __init__(self, threshold = None):
        self._contracts = [
            ((), ('verifier:gt-vs-test:object-detection',)),
            (
                ('sample:ground-truth', 'sample:testing'),
                ('assessment:gt-vs-test:object-detection',),
            ),
        ]
        if threshold == None:
            threshold = finfo(float32).eps
        self._threshold = threshold
    def static_contracts(self):
        return self._contracts
    def get_contract(self, mode_id):
        return self._contracts[mode_id]
    def setup(self, mode_id, input_types, output_mask):
        # there is only one output resource (assessment), if it is marked
        # as not-required then we can return with nothing right away
        if output_mask[0] == False:
            return lambda *args: None, (None,)
        # check simpler mode 0 which is used to present the object itself
        if mode_id == 0:
            return lambda: self, (ResourceTypeInfo(self.__class__, vrf_obj = self),)
        # from now on we consider ourselves in mode 1: do assessment;
        # on input we have two arguments: gt data and test data (single sample)
        gt_cls, ts_cls = map(lambda x: x.type_cls, input_types)
        get_gt_bbox = bounding_box_capability(gt_cls)
        get_ts_bbox = bounding_box_capability(ts_cls)
        # only get_bbox-functions are mandatory, everything else is optional
        # but will be automatically attached when available
        format_dict = {'threshold': self._threshold}
        vars_to_inject = [
            ('find_best_iou', find_best_iou),
            ('empty', empty),
            ('argmax', argmax),
            ('get_gt_bbox', get_gt_bbox),
            ('get_ts_bbox', get_ts_bbox),
        ]
        try:
            get_confidences = confidence_capability(ts_cls)
            output_class = DetectionConfidenceAssessment
            vars_to_inject += [
                ('get_confidences', get_confidences),
                ('flatnonzero', flatnonzero),
                ('OutputClass', output_class),
                ('apply_along_axis', apply_along_axis),
            ]
            format_dict['process_fp_indices'] = \
                'fp_indices = flatnonzero(indices == -1)'
            format_dict['optional_conf_by_idx_setup'] = \
                'conf_by_idx = get_confidences(ts_data).__getitem__'
            format_dict['create_result_object'] = \
                'OutputClass(' \
                    'apply_along_axis(conf_by_idx, 0, tp_indices[:tp_count]),' \
                    'apply_along_axis(conf_by_idx, 0, fp_indices),' \
                    'fn_count,' \
                    'multicount,' \
                ')'
        except: # FIXME
            output_class = DetectionSimpleAssessment
            vars_to_inject += [
                ('count_nonzero', count_nonzero),
                ('OutputClass', output_class),
            ]
            format_dict['process_fp_indices'] = \
                'fp_count = count_nonzero(indices == -1)'
            format_dict['create_result_object'] = \
                'OutputClass(' \
                    'tp_count,' \
                    'fp_count,' \
                    'fn_count,' \
                    'multicount,' \
                ')'
        try:
            # NOTE: flatnonzero is already injected
            get_whitelist = whitelist_capability(gt_cls)
            vars_to_inject.append(('get_whitelist', get_whitelist))
            format_dict['base_indices_genexpr'] = \
                'flatnonzero(get_whitelist(gt_data))'
        except: # FIXME
            format_dict['base_indices_genexpr'] = \
                'range(gt_boxes.shape[0])'
        processor_code = \
            'def worker(gt_data, ts_data):\n' \
            '    gt_boxes = get_gt_bbox(gt_data)\n' \
            '    ts_boxes = get_ts_bbox(ts_data)\n' \
            '    indices, scores = find_best_iou(gt_boxes, ts_boxes)\n' \
            '    indices[scores < {threshold}] = -1\n' \
            '    tp_indices = empty(gt_boxes.shape[1], order = "C", dtype = "int32")\n' \
            '    tp_count = fn_count = multicount = 0\n' \
            '    for idx in {base_indices_genexpr}:\n' \
            '        ts_match_indices = indices == idx\n' \
            '        if ts_match_indices.any():\n' \
            '            iou_scores = scores[ts_match_indices]\n' \
            '            multicount += (iou_scores.shape[0] - 1)\n' \
            '            tp_indices[tp_count] = argmax(iou_scores)\n' \
            '            tp_count += 1\n' \
            '        else:\n' \
            '            fn_count += 1\n' \
            '    {process_fp_indices}\n' \
            '    {optional_conf_by_idx_setup}\n' \
            '    return {create_result_object}'
        wrapped_code = \
            'def make_worker({var_names}):\n' + \
            '\n'.join(map(lambda l: '    ' + l, processor_code.split('\n'))) + \
            '\n    return worker'
        format_dict['var_names'] = ', '.join(map(itemgetter(0), vars_to_inject))
        wrapped_code = wrapped_code.format(**format_dict)
        globs = {}
        exec(compile(wrapped_code, '', 'exec', optimize = 2), {}, globs)
        return (
            globs['make_worker'](*map(itemgetter(1), vars_to_inject)),
            (ResourceTypeInfo(output_class),),
        )
