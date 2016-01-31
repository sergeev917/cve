from .Role import (
    GroundTruthDataset,
    EvaluatedDataset,
)
from .Verifier import IVerifier
from .Driver import IEvaluationDriver
from .Plugin import IPlugin


def evaluate(*args, **kwargs):
    '''Evaluates the given test dateset against the ground truth one'''
    # workflow state class which will do all the work after being configured
    class Evaluator:
        def __init__(self):
            # a queue of actions to be performed: (func, args, kwargs)
            self.queue = []
            # a dictionary where all information is stored
            self.workspace = {}
        def do_step(self):
            func, args, kwargs = self.queue.pop(0)
            func(self.workspace, *args, **kwargs)
        def execute(self):
            while len(self.queue) > 0:
                self.do_step()
    # populating Evaluator instance and setting up default queue
    ev = Evaluator()
    # common searching helper
    def find_and_setup(target_class, target_func, zero_msg, many_msg):
        targets = tuple(x for x in args if isinstance(x, target_class))
        targets_count = len(targets)
        if targets_count == 0:
            raise Exception(zero_msg) # FIXME
        if targets_count > 1:
            raise Exception(many_msg) # FIXME
        ev.queue.append((target_func, targets, {}))
    find_and_setup(
        GroundTruthDataset,
        setup_gt_dataset,
        'ground-truth dataset is missing',
        'multiple ground-truth dataset are not supported',
    )
    find_and_setup(
        EvaluatedDataset,
        setup_eval_dataset,
        'a dataset to be evaluated is missing',
        'multiple datasets to be evaluated are not supported',
    )
    find_and_setup(
        IVerifier,
        setup_verifier,
        'verifier method is missing',
        'multiple verifiers are not supported, take a look at plugins',
    )
    find_and_setup(
        IEvaluationDriver,
        setup_driver,
        'evaluation driver is missing',
        'multiple evaluation drivers are not supported',
    )
    ev.queue.append((do_verifier_config, [], {}))
    ev.queue.append((do_driver_config, [], {}))
    # at this point all configuration is done, inserting the actual work
    ev.queue.append((do_verifier_pass, [], {}))
    ev.queue.append((do_driver_collect, [], {}))
    ev.queue.append((remove_verifier_storage, [], {}))
    ev.queue.append((do_driver_finalize, [], {}))
    ev.queue.append((remove_driver_collect_storage, [], {}))
    # the last thing remains -- patching plugins in:
    # a plugin do observe the entire jobs queue and is free to modify it
    plugins = (x for x in args if isinstance(x, IPlugin))
    for plugin in plugins:
        plugin.inject(ev)
    # doing all the work and returning the driver output
    ev.execute()
    return ev.workspace['out:driver'] # FIXME

# standard jobs to be inserted into task-queue are defined here:
# additional functionality could be inserted via plugins
def setup_gt_dataset(workspace, role_dataset):
    workspace['cfg:gt-dataset'] = role_dataset.dataset

def setup_eval_dataset(workspace, role_dataset):
    workspace['cfg:eval-dataset'] = role_dataset.dataset

def setup_verifier(workspace, verifier):
    workspace['cfg:verifier'] = verifier

def setup_driver(workspace, driver):
    workspace['cfg:driver'] = driver

def do_verifier_config(workspace):
    gt_dataset = workspace['cfg:gt-dataset']
    eval_dataset = workspace['cfg:eval-dataset']
    workspace['cfg:verifier'].reconfigure(gt_dataset, eval_dataset)

def do_driver_config(workspace):
    verifier = workspace['cfg:verifier']
    workspace['cfg:driver'].reconfigure(verifier)

def do_verifier_pass(workspace):
    gt_dataset = workspace['cfg:gt-dataset']
    ev_dataset = workspace['cfg:eval-dataset']
    verify = workspace['cfg:verifier']
    # temporary storage for verifier responses
    storage = []
    # iterating over ground-truth dataset as it has to declare
    # annotation for each sample (including empty annotation), whereas
    # tested annotation format could omit samples with no annotation.
    for sample_name, gt_annotation in gt_dataset:
        ev_annotation = ev_dataset[sample_name]
        if ev_annotation == None:
            ev_annotation = ev_dataset.storage_class()
        storage.append(verify(gt_annotation, ev_annotation))
    workspace['out:verifier'] = storage

def remove_verifier_storage(workspace):
    del workspace['out:verifier']

def do_driver_collect(workspace):
    # NOTE: we could save state after `collect` in the driver instance
    #       itself but we want to keep driver instance reentrant
    # NOTE: `collect` is using the entire output of verifier (and not
    #       per-element) because we need to know (for example) how much
    #       memory to preallocate inside `collect`
    storage = workspace['out:verifier']
    collect = workspace['cfg:driver'].collect
    workspace['out:driver-collect'] = collect(storage)

def remove_driver_collect_storage(workspace):
    del workspace['out:driver-collect']

def do_driver_finalize(workspace):
    finalize = workspace['cfg:driver'].finalize
    collected = workspace['out:driver-collect']
    workspace['out:driver'] = finalize(collected)

