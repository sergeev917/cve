__all__ = (
    'evaluate',
)

from .Logger import get_default_logger
from .FlowBuilder import FlowBuilder

# FIXME: add FlowBuilder option 'keep all data'

def evaluate(*args, **kwargs):
    '''Constructs information flow for the given driver and evaluates it.'''
    logger = kwargs.pop('logger', get_default_logger())
    user_targets = kwargs.get('add_targets', None)
    # FIXME: requested_only option?
    efilter = kwargs.get('hide_resources', None)
    # we are going to provider a default resource-name filter; by default we
    # will hide any output resources which have a form of '__.*__'; the form
    # is used for special cases like side-effect name to enforce node usage
    # by the flow engine
    if efilter is None:
        is_special = lambda n: n.startswith('__') and n.endswith('__')
        # having user-specified targets, exclude them from filtering
        if user_targets is not None:
            def efilter(name):
                return is_special(name) and name not in user_targets
        else:
            efilter = is_special
    flow = FlowBuilder()
    injected_targets = []
    for node in args:
        flow.register(node)
        try:
            injected_targets += node.targets_to_inject()
        except AttributeError:
            pass
    # build all targets: user-given and implicitly injected (side-effects, etc)
    effective_targets = injected_targets + (user_targets or [])
    func_and_types_options = flow.construct(effective_targets)
    if len(func_and_types_options) == 0:
        raise RuntimeError('resource graph construction has failed')
    elif len(func_and_types_options) > 1:
        raise RuntimeError('ambiguous resource graph configuration')
    # selecting the first (and only) option, selecting function and call it
    outdata = func_and_types_options[0][0]()
    # removing service names from output (or applying user-defined filter)
    results = [x for x in zip(effective_targets, outdata) if not efilter(x[0])]
    return dict(results)
