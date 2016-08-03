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
    resource_filter = kwargs.get('hide_resources', None)
    # we are going to provider a default resource-name filter; by default we
    # will hide any output resources which have a form of '__.*__'; the form
    # is used for special cases like side-effect name to enforce node usage
    # by the flow engine
    if resource_filter is None:
        is_special = lambda n: n.startswith('__') and n.endswith('__')
        # having user-specified targets, exclude them from filtering
        if user_targets is not None:
            def resource_filter(name):
                return is_special(name) and name not in user_targets
        else:
            resource_filter = is_special
    flow = FlowBuilder()
    for node in args:
        flow.register(node)
    outnames, execute_func = flow.construct(user_targets)
    if len(execute_func) == 0:
        raise RuntimeError('resource graph construction has failed')
    elif len(execute_func) > 1:
        raise RuntimeError('ambiguous resource graph configuration')
    outdata = execute_func[0]()
    # filtering output
    results = [x for x in zip(outnames, outdata) if not resource_filter(x[0])]
    return dict(results)
