__all__ = (
    'TuneBBoxesAnnotations',
    'TuneProxyDataset',
)

from operator import itemgetter
from numpy import empty_like
from ..Base import (
    IPlugin,
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
from ..Annotation.Sample import BoundingBoxSampleAnnotation

class TuneBBoxesAnnotations(IPlugin):
    '''Applies global scale and translate transform to annotations

    The given setup_idx index must point to a configuration function which
    does setup of dataset to be replaced. The arguments of that function will
    be modified in the way to plug in a proxy dataset which does annotation
    transformation.'''
    def __init__(self, **opts):
        self.rounds = opts.get('rounds', 5)
    def inject(self, ev):
        queue = ev.queue
        # checking for suitable queue: need to find gt/eval datasets
        qfunc = tuple(map(itemgetter(0), queue))
        gt  = [i for i, v in enumerate(qfunc) if v is setup_gt_dataset]
        evals  = [i for i, v in enumerate(qfunc) if v is setup_eval_dataset]
        if len(gt) != 1 or len(evals) != 1:
            raise Exception('gt/eval datasets binding failed') # FIXME
        target_idx = evals[0]
        ins_idx = max(gt[0], target_idx) + 1
        # origin dataset is the one which configured by `setup_eval_dataset`
        origin_dataset = queue[target_idx][1]
        proxy_dataset = TuneProxyDataset(origin_dataset)
        # patching `setup_eval_dataset` step to select our dataset
        queue[target_idx] = (setup_eval_dataset, [proxy_dataset], {})
        verifier = TuneBBoxesVerifier()
        driver = TuneBBoxesDriver()
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

def adjust_transformation():
    pass

class TuneBBoxesDriver(IEvaluationDriver):
    pass

class TuneBBoxesVerifier(IVerifier):
    '''Matches bboxes with good enough iou, returns the best transform.

    The transform coefficients are calculated here per single detection,
    results will be averaged in the driver, which will collect this verifier
    outputs.'''
    def __init__(self):
        pass
    def reconfigure(self, gt_dataset, eval_dataset):
        pass
    def __call__(self, base_sample, test_sample):
        pass

class Transformer:
    def __init__(self, x_shift = 0., y_shift = 0., x_scale = 1., y_scale = 1.):
        self.x_shift, self.y_shift = x_shift, y_shift
        self.x_scale, self.y_scale = x_scale, y_scale
    def __call__(self, bbox_annotation_object):
        numpy_bboxes = bbox_annotation_object.value
        newann = BoundingBoxSampleAnnotation()
        newann.value = empty_like(numpy_bboxes)
        transformed = newann.value
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
        return newann

class TuneProxyDataset(IDatasetAnnotation):
    '''Proxy dataset which does geometric transformation of every sample.

    The transformation includes translation and scale changes independently
    for each axis of 2d annotation. This proxy class replaces only bounding
    boxes information.'''
    # FIXME: a bbox adapter might be left untouched here
    def __init__(self, origin_dataset, **kwargs):
        self.origin = origin_dataset
        transformer_closure = Transformer()
        class TuneProxySample:
            __slots__ = ('origin',)
            def __init__(self, origin):
                self.origin = origin
            def __getattribute__(self, name):
                origin = object.__getattribute__(self, 'origin')
                def_value = origin.__getattribute__(name)
                if name == 'numpy_bounding_boxes':
                    return transformer_closure(def_value)
                return def_value
        self.wrapper = TuneProxySample
        # NOTE: binding to the same transformer that is used in TuneProxySample,
        #       only access it to change -- not rebind/assign to other values
        #       since then TuneProxyDataset and TuneProxySample will use
        #       different objects.
        self.transformer = transformer_closure
    def __getitem__(self, sample_name):
        value = self.origin[sample_name]
        return self.wrapper(value)
    def __iter__(self):
        for name, value in self.origin:
            yield name, self.wrapper(value)
