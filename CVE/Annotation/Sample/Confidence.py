__all__ = ()

from numpy import (
    ndarray,
    append,
)

class ConfidenceSampleAnnotation:
    __slots__ = ('confidence',)
    signature_name = 'numpy_confidences'

    def __init__(self):
        self.confidence = ndarray(0, dtype = 'int32', order = 'C')

    def push(self, new_confidence_value):
        self.confidence = append(self.confidence, new_confidence_value)

class ConfidenceFieldAdapter:
    field_names = ('confidence',)
    def create_annotation(self):
        return ConfidenceSampleAnnotation()

    def __init__(self, field_names):
        # FIXME don't forget to insert append_annotation function
        pass
