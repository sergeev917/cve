__all__ = ('IgnoreAdapter',)

class IgnoreAdapter:
    '''This adapter class is actually just special class which should
    be filtered out on some stage of processing. For example, MutableAnnotation
    will silently drop IgnoreAdapter from the passed adapters list. Note that it
    is possible to create IgnoreAdapter instance (to avoid user-side filtering,
    because a user might want to build adapter instances list to feed it to
    MutableAnnotation) but IgnoreAdapter should not be used far than that.'''
    __slots__ = ()
    storage_class = type(None)
    handled_fields = ('ignore',)
    def __init__(self, *args, **kwargs):
        pass
    def create_annotation(self, *args, **kwargs):
        raise RuntimeError('IgnoreAdapter is actually being used')
    def append_annotation(self, *args, **kwargs):
        raise RuntimeError('IgnoreAdapter is actually being used')
