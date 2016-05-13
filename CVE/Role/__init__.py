__all__ = (
    'GroundTruthDataset',
    'EvaluatedDataset',
    'asGtDataset',
    'asEvalDataset',
)

from ..Base import IDatasetAnnotation

# FIXME: rename dataset with _dataset after migration

class GroundTruthDataset:
    '''An object wrapper which indicates that the wrapped object is GT'''
    __slots__ = ('dataset',)
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The dataset must inherit from IDatasetAnnotation') # FIXME
        # we have a single mode: require nothing and provide dataset/gt
        DependencyFlowNode.__init__(self, [([], ['dataset/ground-truth'])])
        self.dataset = dataset
    def mode_output_getter(self, mode_index):
        # assuming that the given index is correct
        return lambda: (self.dataset,)
    def configure(self, mode_index, input_classes):
        return (self.dataset.__class__,)

class EvaluatedDataset:
    '''An object wrapper which indicates that the wrapped object is tested'''
    __slots__ = ('dataset',)
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The dataset must inherit from IDatasetAnnotation') # FIXME
        # we have a single mode: require nothing and provide dataset/eval
        DependencyFlowNode.__init__(self, [([], ['dataset/evaluated'])])
        self.dataset = dataset
    def mode_output_getter(self, mode_index):
        # assuming that the given index is correct
        return lambda: (self.dataset,)
    def configure(self, mode_index, input_classes):
        return (self.dataset.__class__,)

def asGtDataset(dataset):
    return GroundTruthDataset(dataset)

def asEvalDataset(dataset):
    return EvaluatedDataset(dataset)
