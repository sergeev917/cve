__all__ = ('IgnoreLoader',)

from ...Base import NonApplicableLoader

class IgnoreLoader:
    '''This adapter class is actually just special class which should
    be filtered out on some stage of processing. For example, MutableAnnotation
    will silently drop IgnoreAdapter from the passed adapters list. Note that it
    is possible to create IgnoreAdapter instance (to avoid user-side filtering,
    because a user might want to build adapter instances list to feed it to
    MutableAnnotation) but IgnoreAdapter should not be used far than that.'''
    annotation_class = None
    def setup_from_fields(self, field_names):
        en_names = enumerate(field_names)
        indices = [e[0] for e in enumerate(field_names) if e[1] == 'ignore']
        if len(indices) == 0:
            raise NonApplicableLoader()
        return indices, None
