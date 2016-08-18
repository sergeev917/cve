__all__ = (
    'ConfidenceSampleAnnotation',
    'ConfidenceLoader',
)

from operator import itemgetter
from numpy import (
    ndarray,
    append,
    float32,
)
from ...Base import NonApplicableLoader

class ConfidenceSampleAnnotation:
    __slots__ = ('value', 'top')
    storage_signature = 'std_scores'
    def __init__(self, prealloc = 0):
        self.value = ndarray(prealloc, dtype = 'float32', order = 'C')
        self.top = 0
    def add_record(self, value):
        if self.top < self.value.shape[0]:
            self.value[self.top] = value
        else:
            self.value = append(self.value, value)
        self.top += 1

class ConfidenceLoader:
    annotation_class = ConfidenceSampleAnnotation
    def setup_from_fields(self, field_names):
        indices = [e[0] for e in enumerate(field_names) if e[1] == 'confidence']
        if len(indices) != 1:
            raise NonApplicableLoader()
        def adapter(field_value):
            return float32(field_value)
        return indices, adapter
