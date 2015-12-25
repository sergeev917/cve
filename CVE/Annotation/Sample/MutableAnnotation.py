__all__ = ('generate_sample_annotation_class',)

# NOTE: maybe it would be great to pass arguments to create_annotation()
def generate_sample_annotation_class(adapters):
    # to enforce immutablity as we will use the variable later (closure)
    adapters = tuple(adapters)
    values = tuple(map(lambda adp: adp.create_annotation(), adapters))
    names = tuple(map(lambda val: val.__class__.signature_name, values))
    if len(names) != len(set(names)):
        raise Exception('member names clash') #FIXME
    # generating sample annotation which will include all atomic annotations:
    # for example, the resulting sample annotation could include bounding boxes
    # and confidence annotations. The last ones are accessible as members with
    # signature_name names.
    class MutableSampleAnnotation:
        __slots__ = names
        def __init__(self):
            for name_value_pair in zip(names, values):
                setattr(self, *name_value_pair)
        def broadcast_annotation(self, *args):
            # broadcasting the received arguments to every atomic annotation
            for member_name, adapter in zip(names, adapters):
                adapter.append_annotation(getattr(self, member_name), *args)
    return MutableSampleAnnotation
