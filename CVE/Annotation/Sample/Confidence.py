__all__ = (
    'ConfidenceSampleAnnotation',
    'ConfidenceFieldAdapter',
)

from operator import itemgetter
from numpy import (
    ndarray,
    append,
    float32,
)

class ConfidenceSampleAnnotation:
    __slots__ = ('value',)
    signature_name = 'numpy_confidences'

    def __init__(self):
        self.value = ndarray(0, dtype = 'float32', order = 'C')

    def push(self, confidence_value):
        self.value = append(self.value, confidence_value)

class ConfidenceFieldAdapter:
    storage_class = ConfidenceSampleAnnotation
    handled_fields = ('confidence',)

    def create_annotation(self):
        return self.__class__.storage_class()

    def __init__(self, field_names):
        required_field = 'confidence'
        if required_field not in field_names:
            raise Exception() # FIXME
        pickup_func = itemgetter(field_names.index(required_field))
        def append_annotation(annotation_object, *field_values):
            annotation_object.push(float32(pickup_func(field_values)))
        self.append_annotation = append_annotation
