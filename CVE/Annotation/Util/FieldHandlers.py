__all__ = ('get_global_registry',)
from copy import deepcopy

from ..Sample.BoundingBox import BoundingBoxLoader
from ..Sample.Confidence import ConfidenceLoader
from ..Sample.Ignore import IgnoreLoader
from ..Sample.Whitelist import WhitelistLoader

_registry = (
    BoundingBoxLoader(),
    ConfidenceLoader(),
    IgnoreLoader(),
    WhitelistLoader(),
)

def get_global_registry():
    return deepcopy(_registry)
