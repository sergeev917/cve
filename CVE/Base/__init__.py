__author__  = 'Alexander E. Sergeev'
__contact__ = 'sergeev917@gmail.com'

__all__ = (
    'IEvaluationDriver',
    'UnrecognizedAnnotationFormat',
    'ViolatedAnnotationFormat',
    'IDatasetAnnotation',
    'DetectionSimpleAssessment',
    'DetectionConfidenceAssessment',
)

from operator import itemgetter
from collections import namedtuple

class UnrecognizedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class ViolatedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

DetectionConfidenceAssessment = \
    namedtuple(
        'DetectionConfidenceAssessment',
        ['tp_confs', 'fp_confs', 'fn_count', 'multicount'],
    )

DetectionSimpleAssessment = \
    namedtuple(
        'DetectionSimpleAssessment',
        ['tp_count', 'fp_count', 'fn_count', 'multicount'],
    )

def _raise_not_implemented(obj, function_name):
    derived_class_name = obj.__class__.__qualname__
    message = '"{}" class has no implementation of "{}"'.format(
        derived_class_name,
        function_name,
    )
    raise NotImplementedError(message)

class IDatasetAnnotation:
    '''Represents interface of a dataset annotation class.

    This class is supposed to be used as a base class for specialized
    implementations designed for particular formats of ground-truth
    data or output of object-detection algorithms. Note that IDatasetAnnotation
    does not inherit DependencyFlowNode since the role of a plain dataset class
    is not defined. See CVE.Roles for the further information.'''

    __slots__ = () # since this class won't have any attribute fields

    def __init__(self, path):
        '''Loads annotation from given path.'''
        _raise_not_implemented(self, '__init__')
    def __getitem__(self, sample_name):
        _raise_not_implemented(self, '__getitem__')
    def __setitem__(self, sample_name, new_value):
        _raise_not_implemented(self, '__setitem__')
    def __iter__(self):
        _raise_not_implemented(self, '__iter__')

class IEvaluationDriver:
    '''A class for actual evaluation drivers to be inherited from'''
    def collect(self): #FIXME: arguments and docs
        pass
    def finalize(self):
        pass
