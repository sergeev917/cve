__all__ = ('register_handler', 'get_global_registry')

from copy import deepcopy

registry = dict()

def register_handler(field_name, handler_class):
    global registry
    if field_name in registry:
        raise Exception() # FIXME
    registry[field_name] = handler_class

def get_global_registry():
    return deepcopy(registry)

# registering all implemented handlers
from ..Sample.BoundingBox import BoundingBoxFieldAdapter
for field_name in BoundingBoxFieldAdapter.field_names:
    register_handler(field_name, BoundingBoxFieldAdapter)
