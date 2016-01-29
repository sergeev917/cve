__all__ = ('generate_sample_annotation_class',)

from .Ignore import IgnoreAdapter

# NOTE: maybe it would be great to pass arguments to create_annotation()
# NOTE: "adapters" is a list of *instances* of some adapter classes
def generate_sample_annotation_class(adapters):
    # to enforce immutability as we will use the variable later (closure)
    adapters = tuple(x for x in adapters if not isinstance(x, IgnoreAdapter))
    # need to know signature_name's to build __slots__ for the generated class,
    # to reduce requirements (such we need to get annotation instances:
    names = tuple(map(lambda a: a.storage_class.signature_name, adapters))

    if len(names) != len(set(names)):
        raise Exception('member names clash') #FIXME
    # generating sample annotation which will include all atomic annotations:
    # for example, the resulting sample annotation could include bounding boxes
    # and confidence annotations. The last ones are accessible as members with
    # signature_name names.
    class MutableSampleAnnotation:
        __slots__ = names
        def __init__(self):
            for member_name, adapter in zip(names, adapters):
                setattr(self, member_name, adapter.create_annotation())
        def push_all(self, *args):
            # broadcasting the received arguments to every sub-annotation
            for member_name, adapter in zip(names, adapters):
                adapter.append_annotation(getattr(self, member_name), *args)
    return MutableSampleAnnotation
