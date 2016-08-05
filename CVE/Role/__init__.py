__all__ = (
    'GroundTruthDataset',
    'EvaluatedDataset',
    'asGtDataset',
    'asEvalDataset',
)

from ..Base import IDatasetAnnotation
from ..FlowBuilder import DataInjectorNode, ResourceTypeInfo

# FIXME: rename dataset with _dataset after migration
# FIXME: remove dependency from IDatasetAnnotation if it does not implement
#        some base stuff

class GroundTruthDataset(DataInjectorNode):
    '''An object wrapper which indicates that the wrapped object is GT'''
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The dataset must inherit from IDatasetAnnotation') # FIXME
        type_info = ResourceTypeInfo(
            type(dataset),
            storage_class = dataset.storage_class,
        )
        resources = {'dataset:ground-truth': (dataset, type_info)}
        DataInjectorNode.__init__(self, resources, types_provided = True)

class EvaluatedDataset(DataInjectorNode):
    '''An object wrapper which indicates that the wrapped object is tested'''
    def __init__(self, dataset):
        if not isinstance(dataset, IDatasetAnnotation):
            raise Exception('The dataset must inherit from IDatasetAnnotation') # FIXME
        type_info = ResourceTypeInfo(
            type(dataset),
            storage_class = dataset.storage_class,
        )
        resources = {'dataset:testing': (dataset, type_info)}
        DataInjectorNode.__init__(self, resources, types_provided = True)

def asGtDataset(dataset):
    return GroundTruthDataset(dataset)

def asEvalDataset(dataset):
    return EvaluatedDataset(dataset)
