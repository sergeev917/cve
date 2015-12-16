'''Annotation.Base: base classes for dataset annotations classes.'''
__author__  = 'Alexander E. Sergeev'
__contact__ = 'sergeev917@gmail.com'

__all__ = (
    'UnrecognizedAnnotationFormat',
    'ViolatedAnnotationFormat',
    'DatasetAnnotation',
)

class UnrecognizedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class ViolatedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class DatasetAnnotation:
    '''Represents interface of a dataset annotation class.

    This class is supposed to be used as a base class for specialized
    implementations designed for particular formats of ground-truth
    data or output of object-detection algorithms.'''

    __slots__ = () # since this class won't have any attribute fields

    def __init__(self, path):
        '''Loads annotation from given path.'''
        raise NotImplementedError('DatasetAnnotation base class is used')

    def __getitem__(self, sample_name):
        pass

    def __setitem__(self, sample_name, new_value):
        pass

    def __delitem__(self, sample_name):
        pass

    def __iter__(self):
        # use yield
        pass
