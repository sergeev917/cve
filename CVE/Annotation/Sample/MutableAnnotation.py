__all__ = ('compose_annotation_class',)

from ...Base import (
    exec_with_injection,
    NonApplicableLoader,
    LoaderMissingData,
)

# FIXME: add default adapter
# NOTE: the function is not thread-safe
def compose_annotation_class(fields, annotations_bank):
    # matching fields and adapters; lookup_buffer is a common storage
    # to lookup adapters, annotation-classes which are referred from
    # available_handlers; the last one tracks possible processors per each
    # field in the input
    lookup_buffer = []
    available_handlers  = [list() for i in range(len(fields))]
    for ann_adapter in annotations_bank:
        try:
            indices, adapter_func = ann_adapter.setup_from_fields(fields)
            handler_pkg = (indices, adapter_func, ann_adapter.annotation_class)
            lookup_buffer.append(handler_pkg)
            assigned_idx = len(lookup_buffer) - 1 # NOTE: race conditions
            for index in indices:
                available_handlers[index].append(assigned_idx)
        except NonApplicableLoader: # FIXME
            continue
    # NOTE: as other option we can force user to specify the required data
    # to be able to use dependency flow on handlers
    if any(len(e) != 1 for e in available_handlers):
        raise RuntimeError('annotation field handler is missing or ambiguous') # FIXME
    # since now each field has a single processor, we dropping list-elements
    available_handlers = set([e[0] for e in available_handlers])
    applied_handlers = [lookup_buffer[i] for i in available_handlers]
    # filtering applied_handlers to exclude ignored fields: for these fields
    # annotation class set to None
    applied_handlers = [e for e in applied_handlers if e[2] is not None]
    # now we are going to build an annotation class from selected handlers;
    # first, we prepare class-level dictionary which maps named-subannotation
    # to the corresponding index
    signatures_list = [e[2].storage_signature for e in applied_handlers]
    signatures_dict = dict((e[::-1] for e in enumerate(signatures_list)))
    # the following string will go into class definition as a class variable
    signatures = repr(signatures_dict)
    # next step is to produce class instance init lines
    def idx_pick(indices):
        # note that with one index the input is not a list but just value!
        if len(indices) == 1:
            return 'v[{}]'.format(indices[0])
        # checking for slicing availability
        step, rem = divmod((indices[-1] - indices[0]), len(indices) - 1)
        if rem == 0:
            idx_range = range(indices[0], indices[-1] + 1, step)
            if all((e[0] == e[1] for e in zip(indices, idx_range))):
                # indices form a range, so we can more easily extract fields
                return 'v[{0.start}:{0.stop}:{0.step}]'.format(idx_range)
        return '[v[i] for i in [{}]]'.format(','.join(indices))
    inject_data = [
        ('cls_list', [e[2] for e in applied_handlers]),
        ('cnv', [e[1] for e in applied_handlers]),
    ]
    format_data = {
        'signatures': signatures,
        'tuple_init': 'tuple((cls(cnt) for cls in cls_list))',
        'distrib_data':
            'for e in zip(self, cnv, [{}]): e[0].add_record(e[1](e[2]))'.format(
                ','.join([idx_pick(e[0]) for e in applied_handlers]),
            )
    }
    class_code = \
        'class ComposedSampleAnnotation(tuple):\n' \
        '    __slots__ = ()\n' \
        '    signatures = {signatures}\n' \
        '    def __new__(cls, cnt):\n' \
        '        return tuple.__new__(cls, {tuple_init})\n' \
        '    def add_record(self, v):\n' \
        '        {distrib_data}\n'
    class_code = class_code.format(**format_data)
    return exec_with_injection(
        class_code,
        'ComposedSampleAnnotation',
        inject_data,
    )
