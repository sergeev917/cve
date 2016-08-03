__all__ = (
    'SimpleWalker',
)

from ..FlowBuilder import (
    FlowBuilder,
    PureStaticNode,
    DataInjectorNode,
)
            # put output name into verifier name? like:
            # verifier:gt-vs-test:object-detections
            # and output like:
            # assessment-list:object-detections

# NOTE: not safe for parallel use, race conditions on storing new mode id

class SimpleWalker:
    def __init__(self, **kwargs):
        self._recorded_modes = []
        self._recorded_mode_kind = []
        self._saved_replies = {}
    def dynamic_contracts(self, target_name, present_resources):
        required_prefix = 'assessment-list:'
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
            expected_idx = len(self._recorded_modes)
            self._recorded_modes.append((requires, provides))
            self._recorded_mode_kind.append(mode_kind_id)
            mode_ids_reply.append(expected_idx)
        self._saved_replies[target_name] = mode_ids_reply
        return list(mode_ids_reply)
    def get_contract(self, mode_id):
        return self._recorded_modes[mode_id]
    def setup(self, mode_id, input_types, output_mask):
        mode_processors = [
            self._setup_gt_mode,
        ]
        return mode_processors[mode_id](input_types, output_mask)
    def _setup_gt_mode(self, input_types, output_mask): # FIXME pass assessment resource name
        dataset_gt_t, dataset_ts_t, verifier_t = input_types
        gt_samples_t = dataset_gt_t.aux_info['sample_type']
        ts_samples_t = dataset_ts_t.aux_info['sample_type']
        # going to break type/instance separation here in order to check
        # sanity of verifier & samples types combination; otherwise, we might
        # agree on some non-working configuration with the problem discovered
        # on running stage (after configuration); also, this might lead to
        # multiple configuration with different verifiers -- and that
        # is usually considered an error (i.e. ambiguity)
        verifier = verifier_t.aux_info['object']
        # use additional flow builder here to connect samples, verifier and
        # verifier outputs; we will need to setup two sample-injection nodes
        class _InjectionNode:
            def __init__(self):
                pass
            def static_contracts(self):
                pass
            def get_contract(self, mode_id):
                pass
        gt_sample = _InjectionNode('sample:ground-truth')
        ts_sample = _InjectionNode('sample:testing')
        flow = FlowBuilder()
        flow.register(gt_sample)
        flow.register(ts_sample)
        flow.register(verifier)
        names, funcs = flow.construct('assessment:gt-vs-test:{}'.format(as_kind))
        def execute(dataset_gt, dataset_tst, verifier):
            pass
        return execute
