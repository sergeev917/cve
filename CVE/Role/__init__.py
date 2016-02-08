__all__ = (
    'GroundTruthDataset',
    'EvaluatedDataset',
    'asGtDataset',
    'asEvalDataset',
)

from ..Base import IDatasetAnnotation

class GroundTruthDataset:
    '''An object wrapper which indicates that the wrapped object is GT'''
    __slots__ = ('dataset',)
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The given dataset must be DatasetAnnotation') # FIXME
        self.dataset = dataset

class EvaluatedDataset:
    '''An object wrapper which indicates that the wrapped object is tested'''
    __slots__ = ('dataset',)
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The given dataset must be DatasetAnnotation') # FIXME
        self.dataset = dataset

def asGtDataset(dataset):
    return GroundTruthDataset(dataset)

def asEvalDataset(dataset):
    return EvaluatedDataset(dataset)
