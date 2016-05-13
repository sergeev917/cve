# NOTE: functions other that evaluate() are exported to ease up a plugin writing
__all__ = (
    'setup_gt_dataset',
    'setup_eval_dataset',
    'setup_verifier',
    'setup_driver',
    'do_verifier_config',
    'do_driver_config',
    'do_verifier_pass',
    'do_driver_collect',
    'evaluate',
)

from .Role import (
    GroundTruthDataset,
    EvaluatedDataset,
)
from .Base import (
    IVerifier,
    IEvaluationDriver,
)
from .Logger import get_default_logger
from .FlowGraph import DependencyFlowManager

def evaluate(*args, **kwargs):
    '''Constructs information flow for the given driver and evaluates it.'''
    logger = kwargs.pop('logger', get_default_logger())
    flow = DependencyFlowManager()
    # here we have only flow nodes and plugins
    for node in args:
        flow.add_node(node)


#def evaluate(*args, **kwargs):
#    '''Evaluates the given test dateset against the ground truth one'''
#    logger = kwargs.pop('logger', get_default_logger())
#    # workflow state class which will do all the work after being configured
#    class Evaluator:
#        def __init__(self):
#            self.logger = logger
#            # a queue of actions to be performed: (func, args, kwargs)
#            self.queue = []
#            # a dictionary where all information is stored
#            self.workspace = {}
#        def do_step(self):
#            func, args, kwargs = self.queue.pop(0)
#            func(self.workspace, self.logger, *args, **kwargs)
#        def execute(self):
#            while len(self.queue) > 0:
#                self.do_step()
#    # populating Evaluator instance and setting up default queue
#    ev = Evaluator()
#    # common searching helper
#    def find_and_setup(target_class, target_func, zero_msg, many_msg):
#        targets = tuple(x for x in args if isinstance(x, target_class))
#        targets_count = len(targets)
#        if targets_count == 0:
#            raise Exception(zero_msg) # FIXME
#        if targets_count > 1:
#            raise Exception(many_msg) # FIXME
#        ev.queue.append((target_func, targets, {}))
#    find_and_setup(
#        GroundTruthDataset,
#        setup_gt_dataset,
#        'ground-truth dataset is missing',
#        'multiple ground-truth dataset are not supported',
#    )
#    find_and_setup(
#        EvaluatedDataset,
#        setup_eval_dataset,
#        'a dataset to be evaluated is missing',
#        'multiple datasets to be evaluated are not supported',
#    )
#    find_and_setup(
#        IVerifier,
#        setup_verifier,
#        'verifier method is missing',
#        'multiple verifiers are not supported, take a look at plugins',
#    )
#    find_and_setup(
#        IEvaluationDriver,
#        setup_driver,
#        'evaluation driver is missing',
#        'multiple evaluation drivers are not supported',
#    )
#    ev.queue.append((do_verifier_config, [], {}))
#    ev.queue.append((do_driver_config, [], {}))
#    # at this point all configuration is done, inserting the actual work
#    ev.queue.append((do_verifier_pass, [], {}))
#    ev.queue.append((do_driver_collect, [], {}))
#    # the last thing remains -- patching plugins in:
#    # a plugin do observe the entire jobs queue and is free to modify it
#    plugins = (x for x in args if isinstance(x, IPlugin))
#    for plugin in plugins:
#        plugin.inject(ev)
#    # now doing all the planned work
#    ev.execute()
#    # cleaning the workspace up
#    for key in list(ev.workspace.keys()):
#        if not key.startswith('export:'):
#            del ev.workspace[key]
#    return ev.workspace

# standard jobs to be inserted into task-queue are defined here:
# additional functionality could be inserted via plugins
def setup_gt_dataset(workspace, logger, role_dataset):
    workspace['env:dataset:gt'] = role_dataset.dataset

def setup_eval_dataset(workspace, logger, role_dataset):
    workspace['env:dataset:eval'] = role_dataset.dataset

def setup_verifier(workspace, logger, verifier):
    workspace['env:verifier'] = verifier

def setup_driver(workspace, logger, driver):
    workspace['env:driver'] = driver

def do_verifier_config(workspace, logger):
    gt_dataset = workspace['env:dataset:gt']
    eval_dataset = workspace['env:dataset:eval']
    workspace['env:verifier'].reconfigure(gt_dataset, eval_dataset)

def do_driver_config(workspace, logger):
    verifier = workspace['env:verifier']
    workspace['env:driver'].reconfigure(verifier)

def do_verifier_pass(workspace, logger):
    gt_dataset = workspace['env:dataset:gt']
    ev_dataset = workspace['env:dataset:eval']
    verify = workspace['env:verifier']
    # temporary storage for verifier responses
    storage = []
    # iterating over ground-truth dataset as it has to declare
    # annotation for each sample (including empty annotation), whereas
    # tested annotation format could omit samples with no annotation.
    verifier_name = verify.__class__.__qualname__
    subtask_name = 'verifying dataset samples with "{}"'.format(verifier_name)
    def make_pass(ticker = None):
        for sample_name, gt_annotation in gt_dataset:
            ev_annotation = ev_dataset[sample_name]
            if ev_annotation == None:
                ev_annotation = ev_dataset.storage_class()
            storage.append(verify(gt_annotation, ev_annotation))
            ticker() if ticker else None # ticking the current sample
    if logger:
        gt_dataset_size = len(gt_dataset)
        with logger.progress(subtask_name, gt_dataset_size) as spinner:
            make_pass(spinner.tick)
    else:
        make_pass()
    workspace['tmp:verifier:pass'] = storage

def do_driver_collect(workspace, logger):
    # NOTE: `collect` is using the entire output of verifier (and not
    #       per-element) because we need to know (for example) how much
    #       memory to preallocate inside `collect`
    storage = workspace['tmp:verifier:pass']
    driver = workspace['env:driver']
    driver_name = driver.__class__.__qualname__
    msg = 'processing verifier output with "{}"'.format(driver_name)
    with logger.subtask(msg) as subtask_logger:
        workspace['export:driver'] = driver.collect(storage, subtask_logger)
