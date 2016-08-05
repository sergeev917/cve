__all__ = (
    'SimpleWalker',
)

from ..FlowBuilder import (
    FlowBuilder,
    PureStaticNode,
    DataInjectorNode,
    ResourceTypeInfo,
)

# NOTE: not thread-safe, race conditions on storing new mode id

class SimpleWalker:
    def __init__(self, **kwargs):
        self._recorded_modes = []
        self._recorded_mode_kind = []
        self._saved_replies = {}
    def dynamic_contracts(self, target_name, present_resources):
        required_prefix = 'assessment-list:'
        print(target_name)
        if not target_name.startswith(required_prefix):
            return []
        saved_reply = self._saved_replies.get(target_name, None)
        if saved_reply is not None:
            return saved_reply
        assessment_kind = target_name[len(required_prefix):]
        needed_verifier_pattern = ':'.join(['verifier', '{}', assessment_kind])
        mode_ids_reply = []
        for mode_kind_id, mode_name in enumerate(['gt-vs-test']):
            verifier = needed_verifier_pattern.format(mode_name)
            requires = ('dataset:ground-truth', 'dataset:testing', verifier)
            provides = (target_name,)
            expected_idx = len(self._recorded_modes) # not thread-safe
            self._recorded_modes.append((requires, provides))
            self._recorded_mode_kind.append((mode_kind_id, assessment_kind))
            mode_ids_reply.append(expected_idx)
        self._saved_replies[target_name] = mode_ids_reply
        return list(mode_ids_reply)
    def get_contract(self, mode_id):
        return self._recorded_modes[mode_id]
    def setup(self, mode_id, input_types, output_mask):
        mode_kind_procs = [
            self._setup_gt_mode,
        ]
        mode_kind_id, as_kind = self._recorded_mode_kind[mode_id]
        return mode_kind_procs[mode_kind_id](as_kind, input_types, output_mask)
    def _setup_gt_mode(self, as_kind, input_types, output_mask):
        dataset_gt_t, dataset_ts_t, verifier_t = input_types
        gt_sample_cls = dataset_gt_t.aux_info.get('storage_class', None)
        ts_sample_cls = dataset_ts_t.aux_info.get('storage_class', None)
        if gt_sample_cls is None or ts_sample_cls is None:
            print('walker: no sample_type found')
            raise RuntimeError() # FIXME
        # going to break type/instance separation here in order to check
        # sanity of verifier & samples types combination; otherwise, we might
        # agree on some non-working configuration with the problem discovered
        # on running stage (after configuration); also, this might lead to
        # multiple configuration with different verifiers -- and that
        # is usually considered an error (i.e. ambiguity)
        vrf_obj = verifier_t.aux_info.get('vrf_obj', None)
        if vrf_obj is None:
            print('walker: no verifier object found')
            raise RuntimeError() # FIXME
        # use additional flow builder here to connect samples, verifier and
        # verifier outputs; we will need to setup two sample-injection nodes
        class _InjectionNode:
            __slots__ = ('data', '_contract', '_output_type')
            def __init__(self, resource_name, output_type):
                self._contract = ((), (resource_name,))
                self._output_type = output_type
            def static_contracts(self):
                return [self._contract]
            def get_contract(self, mode_id):
                return self._contract
            def setup(self, mode_id, input_types, output_mask):
                # need to capture "self", since self.data is a placeholder
                return (lambda: self.data, (self._output_type,))
        gt_sample = _InjectionNode('sample:ground-truth', ResourceTypeInfo(gt_sample_cls))
        ts_sample = _InjectionNode('sample:testing', ResourceTypeInfo(ts_sample_cls))
        flow = FlowBuilder()
        flow.register(gt_sample)
        flow.register(ts_sample)
        flow.register(vrf_obj)
        funcs = flow.construct(['assessment:gt-vs-test:{}'.format(as_kind)])
        if len(funcs) > 1:
            raise RuntimeError() # FIXME
        elif len(funcs) == 0:
            raise RuntimeError() # FIXME
        do_assessment, output_type_info = funcs[0]
        def execute(gt_dataset, ts_dataset, verifier):
            # we have already grabbed verifier object via its type information
            # since we need to know (access) the object itself not only its type
            # for being able to put it again into our nested flowbuilder; here,
            # we just make sure that the object from type information is the
            # same as we are given right now (to avoid tricky errors)
            assert verifier is vrf_obj
            # this is the core of dataset walker -- to pick matching pairs
            # of annotations and run the selected verifier on them
            assessment_list = []
            for sample_name, gt_ann in gt_dataset:
                gt_sample.data = gt_ann
                ts_sample.data = ts_dataset[sample_name] # FIXME: add get(None)
                # since construct()-provided functions return list of values,
                # we need to extract the first field (and the only field)
                assessment_list.append(do_assessment()[0])
            return assessment_list
        return execute, (ResourceTypeInfo(list, elem_type = output_type_info),)
