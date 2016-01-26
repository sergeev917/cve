__all__ = (
    'DetectionCurveDriver',
)

from .Driver import IEvaluationDriver
from numpy import empty, argsort

class PrecisionHandler:
    __slots__ = ()
    def __init__(self, verifier, verifier_out_lst):
        pass

class RecallHandler:
    __slots__ = ()
    def __init__(self, verifier, verifier_out_lst):
        pass

class FPCountHandler:
    __slots__ = ()
    def __init__(self):
        pass

class FPPIHandler:
    __slots__ = ()
    def __init__(self):
        pass

available_handlers = {
    'precision': PrecisionHandler,
    'recall': RecallHandler,
    'fp': FPCountHandler,
    'fppi': FPPIHandler,
}

class DetectionCurveDriver(IEvaluationDriver):
    __slots__ = ('x_driver_class', 'y_driver_class', 'verifier')
    def __init__(self, **kwargs):
        y_driver, x_driver = kwargs.get('mode', ('precision', 'recall'))
        if y_driver not in available_handlers:
            raise Exception('unknown notion "{}" is requested'.format(y_driver)) # FIXME
        if x_driver not in available_handlers:
            raise Exception('unknown notion "{}" is requested'.format(x_driver)) # FIXME
        self.x_driver_class = available_handlers[x_driver]
        self.y_driver_class = available_handlers[y_driver]
    def reconfigure(self, verifier_instance):
        if verifier_instance.interpret_as != 'confidence-relative':
            raise Exception('only confidence-relative input is supported') # FIXME
        self.verifier = verifier_instance
    def collect(self, verifier_output):
        # creating X/Y axis value handlers for the given verifier responses
        x_drv = self.x_driver_class(self.verifier, verifier_output)
        y_drv = self.y_driver_class(self.verifier, verifier_output)
        # gathering size of confidence array to create:
        # we will use FP and TP confidence points
        need_conf_size = lambda out: out[0].shape[0] + out[1].shape[0]
        conf_size = sum(map(need_conf_size, verifier_output))
        # preallocating array for confidences and payloads of the required size
        # NOTE: payload is simply `is_true_positive` indicator (for now)
        # NOTE: no numpy.concatenate -- it will require a sequence (list/tuple)
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
        # sorting by confidence to do thesholding on it later (descending)
        # FIXME: check mergesort instead of quicksort
        sort_indices = argsort(confs, kind = 'quicksort')[::-1]
        confs = confs[sort_indices]
        payload = payload[sort_indices]
        # thresholding confidence level from the highest one

    def finalize(self, collection_out):
        pass
