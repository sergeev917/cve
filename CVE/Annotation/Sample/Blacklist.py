__all__ = (
    'BlacklistSampleAnnotation',
    'BlacklistFieldAdapter',
)

from operator import itemgetter
from numpy import (
    ndarray,
    append,
    float32,
)

# FIXME: preallocation for annotation? Could be done with sample info buffering
class BlacklistSampleAnnotation:
    __slots__ = ('value',)
    signature_name = 'numpy_blacklist'
    def __init__(self):
        self.value = ndarray(0, dtype = 'bool', order = 'C')
    def push(self, is_not_blacklisted):
        self.value = append(self.value, is_not_blacklisted)

class BlacklistFieldAdapter:
    storage_class = BlacklistSampleAnnotation
    handled_fields = ('blacklist_01', 'blacklist_tf')

    def create_annotation(self):
        return self.__class__.storage_class()

    def __init__(self, field_names, **opts):
        targets = tuple(filter(lambda f: f in self.handled_fields, field_names))
        if len(targets) == 0:
            raise Exception() # FIXME
        if len(targets) > 1:
            raise Exception() # FIXME
        selected_field = targets[0]
        pickup_func = itemgetter(field_names.index(selected_field))
        converter = {
            'blacklist_01': lambda field: field == '0',
            'blacklist_tf': lambda field: field.upper() == 'F',
        }[selected_field]
        def append_annotation(annotation_object, *field_values):
            annotation_object.push(converter(pickup_func(field_values)))
        self.append_annotation = append_annotation
