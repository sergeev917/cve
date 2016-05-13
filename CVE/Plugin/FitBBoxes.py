__all__ = (
    'TuneBBoxesAnnotations',
    'TuneProxyDataset',
)

from operator import itemgetter
from numpy import (
    empty_like,
    empty,
    argmax,
)
from ..Base import (
    IDatasetAnnotation,
    IEvaluationDriver,
    IVerifier,
)
from ..evaluate import (
    setup_gt_dataset,
    setup_eval_dataset,
    setup_verifier,
    setup_driver,
    do_verifier_config,
    do_driver_config,
    do_verifier_pass,
    do_driver_collect,
)
from ..Annotation.Capability import bounding_box_capability
from ..Annotation.Sample import BoundingBoxSampleAnnotation
from ..Verifier.BoundingBoxCmp import find_best_iou
from ..Logger import get_default_logger

class TuneBBoxesAnnotations:
    '''Applies global scale and translate transform to annotations

    The given setup_idx index must point to a configuration function which
    does setup of dataset to be replaced. The arguments of that function will
    be modified in the way to plug in a proxy dataset which does annotation
    transformation.'''
    def __init__(self, **opts):
        self.rounds = opts.pop('rounds', 5)
        self.opts = opts
    def inject(self, ev, **kwargs):
        if self.rounds == 0:
            return
        queue = ev.queue
        logger = kwargs.get('logger', get_default_logger())
        # checking for suitable queue: need to find gt/eval datasets
        qfunc = tuple(map(itemgetter(0), queue))
        gt  = [i for i, v in enumerate(qfunc) if v is setup_gt_dataset]
        evals  = [i for i, v in enumerate(qfunc) if v is setup_eval_dataset]
        if len(gt) != 1 or len(evals) != 1:
            raise Exception('gt/eval datasets binding failed') # FIXME
        target_idx = evals[0]
        ins_idx = max(gt[0], target_idx) + 1
        # origin dataset is the one which configured by `setup_eval_dataset`
        eval_params, eval_dict_params = queue[target_idx][1:]
        eval_params[0].dataset = TuneProxyDataset(eval_params[0].dataset)
        # patching `setup_eval_dataset` step to select our dataset
        queue[target_idx] = (setup_eval_dataset, eval_params, eval_dict_params)
        verifier = TuneBBoxesVerifier(**self.opts)
        driver = TuneBBoxesDriver(**self.opts)
        setup = [
            (setup_verifier, [verifier], {}),
            (setup_driver, [driver], {}),
            (do_verifier_config, [], {}),
            (do_driver_config, [], {}),
        ]
        single_round = [
            (do_verifier_pass, [], {}),
            (do_driver_collect, [], {}),
            (adjust_transformation, [], {}),
        ]
        inserted_actions = setup + single_round * self.rounds
        ev.queue = queue[:ins_idx] + inserted_actions + queue[ins_idx:]

def adjust_transformation(workspace, logger):
    params = workspace.pop('export:driver')
    g = workspace['env:dataset:eval']._transformer
    g.x_shift, g.y_shift, g.x_scale, g.y_scale = params

class TuneBBoxesDriver(IEvaluationDriver):
    def __init__(self, **kwargs):
        pass
    def reconfigure(self, verifier_instance):
        pass
    def collect(self, verifier_output, logger):
        elements_sum = empty(4, dtype = 'float64')
        elements_sum.fill(0.)
        elements_count = 0
        for sample_result in verifier_output:
            elements_sum += sample_result.sum(axis = 1)
            elements_count += sample_result.shape[1]
        return elements_sum / elements_count

# NOTE: try multi-angle adaptation
class TuneBBoxesVerifier(IVerifier):
    '''Matches bboxes with good enough iou, returns the best transform.

    The transform coefficients are calculated here per single detection,
    results will be averaged in the driver, which will collect this verifier
    outputs.'''
    def __init__(self, **kwargs):
        self.threshold = kwargs.get('threshold', 0.5)
    def reconfigure(self, gt_dataset, eval_dataset):
        gt_storage_class = gt_dataset.storage_class
        eval_storage_class = eval_dataset.storage_class
        self.get_base_bbox = bounding_box_capability(gt_storage_class)
        self.get_test_bbox = bounding_box_capability(eval_storage_class)
    def __call__(self, base_sample, test_sample):
        base_set = self.get_base_bbox(base_sample)
        test_set = self.get_test_bbox(test_sample)
        indices, scores = find_best_iou(base_set, test_set)
        # resetting indices of low-iou matches to '-1'
        indices[scores < self.threshold] = -1
        center = lambda p: (p[0] + p[1]) / 2.
        length = lambda p: float(p[1] - p[0])
        # aggregating delta and scales for matching pairs
        gt_size = base_set.shape[1]
        params = empty((4, gt_size), order = 'C', dtype = 'float32')
        params_filled = 0
        for base_idx in range(0, gt_size):
            match_indices = indices == base_idx
            if match_indices.any():
                match_iou_scores = scores[match_indices]
                selected_idx = argmax(match_iou_scores)
                # now we have a pair: base_idx, selected_idx
                a, b = base_set[:, base_idx], test_set[:, selected_idx]
                a_x, b_x, a_y, b_y = a[0::2], b[0::2], a[1::2], b[1::2]
                params[:, params_filled] = (
                    center(b_x) - center(a_x),
                    center(b_y) - center(a_y),
                    length(b_x) / length(a_x),
                    length(b_y) / length(a_y),
                )
                params_filled += 1
        return params[:, :params_filled]

class Transformer:
    def __init__(self, x_shift = 0., y_shift = 0., x_scale = 1., y_scale = 1.):
        self.x_shift, self.y_shift = x_shift, y_shift
        self.x_scale, self.y_scale = x_scale, y_scale
    def __call__(self, numpy_bboxes):
        transformed = empty_like(numpy_bboxes)
        xk1, xk2 = (self.x_scale + 1.) / 2., (1. - self.x_scale) / 2.
        yk1, yk2 = (self.y_scale + 1.) / 2., (1. - self.y_scale) / 2.
        transformed[0, :] = numpy_bboxes[0, :] * xk1 + \
                            numpy_bboxes[2, :] * xk2 + \
                            self.x_shift
        transformed[2, :] = numpy_bboxes[0, :] * xk2 + \
                            numpy_bboxes[2, :] * xk1 + \
                            self.x_shift
        transformed[1, :] = numpy_bboxes[1, :] * yk1 + \
                            numpy_bboxes[3, :] * yk2 + \
                            self.y_shift
        transformed[3, :] = numpy_bboxes[1, :] * yk2 + \
                            numpy_bboxes[3, :] * yk1 + \
                            self.y_shift
        return transformed

class TuneProxyDataset(IDatasetAnnotation):
    '''Proxy dataset which does geometric transformation of every sample.

    The transformation includes translation and scale changes independently
    for each axis of 2d annotation. This proxy class replaces only bounding
    boxes information.'''
    def __init__(self, origin_dataset, **kwargs):
        origin_storage_class = origin_dataset.storage_class
        self._origin = origin_dataset
        # we need to know bounding box (via some method or stored information)
        get_bbox = bounding_box_capability(origin_storage_class)
        transformer_closure = Transformer()
        class TuneProxySample:
            __proxied_slots__ = origin_storage_class.__slots__
            __slots__ = ('_origin',)
            def __init__(self, origin_sample):
                self._origin = origin_sample
            def bounding_box_capability(self):
                # explicit implementation of the bounding-box capability
                return transformer_closure(get_bbox(self._origin))
            def __getattr__(self, name):
                return self._origin.__getattribute__(name)
        # NOTE: binding to the same transformer that is used in TuneProxySample,
        #       only access it to change -- not rebind/assign to other values
        #       since then TuneProxyDataset and TuneProxySample will use
        #       different objects.
        self._transformer = transformer_closure
        self.storage_class = TuneProxySample
    def __getitem__(self, sample_name):
        value = self._origin[sample_name]
        return self.storage_class(value)
    def __iter__(self):
        for name, value in self._origin:
            yield name, self.storage_class(value)
    def __len__(self):
        return self._origin.__len__()
