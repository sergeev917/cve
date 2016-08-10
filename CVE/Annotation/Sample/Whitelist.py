__all__ = (
    'WhitelistSampleAnnotation',
    'WhitelistLoader',
)

from operator import itemgetter
from numpy import (
    ndarray,
    append,
    float32,
)
from ...Base import (
    NonApplicableLoader,
    LoaderMissingData,
)

class WhitelistSampleAnnotation:
    __slots__ = ('value', 'top')
    storage_signature = 'numpy_whitelist'
    def __init__(self, prealloc = 0):
        self.value = ndarray(prealloc, dtype = 'bool', order = 'C')
        self.top = 0
    def add_record(self, value):
        if self.top < self.value.shape[0]:
            self.value[self.top] = value
        else:
            self.value = append(self.value, value)
        self.top += 1

class WhitelistLoader:
    annotation_class = WhitelistSampleAnnotation
    def setup_from_fields(self, field_names):
        prefix = 'whitelist_'
        idx = [e[0] for e in enumerate(field_names) if e[1].startswith(prefix)]
        if len(idx) != 1:
            raise NonApplicableLoader()
        suffix = field_names[idx[0]][len(prefix):]
        if len(suffix) != 2:
            raise NonApplicableLoader()
        norm, ign = suffix
        def adapter(field_value):
            return field_value == norm
        return idx, adapter
