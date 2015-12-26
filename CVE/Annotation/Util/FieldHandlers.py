__all__ = ('get_global_registry',)

from copy import deepcopy

registry = dict()

def register_handler(field_name, handler_class):
    global registry
    if field_name in registry:
        raise Exception() # FIXME
    registry[field_name] = handler_class

# NOTE: use default value as for "ignore"
def get_global_registry():
    return deepcopy(registry)

# registering all implemented handlers
from ..Sample.BoundingBox import BoundingBoxFieldAdapter
from ..Sample.Confidence import ConfidenceFieldAdapter
from ..Sample.Ignore import IgnoreAdapter
adapters = (
    BoundingBoxFieldAdapter,
    ConfidenceFieldAdapter,
    IgnoreAdapter,
)
for adapter_class in adapters:
    for field_name in adapter_class.handled_fields:
        register_handler(field_name, adapter_class)
