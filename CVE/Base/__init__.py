__author__  = 'Alexander E. Sergeev'
__contact__ = 'sergeev917@gmail.com'

__all__ = (
    'IEvaluationDriver',
    'IPlugin',
    'IVerifier',
    'UnrecognizedAnnotationFormat',
    'ViolatedAnnotationFormat',
    'IDatasetAnnotation',
)

class UnrecognizedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class ViolatedAnnotationFormat(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class IDatasetAnnotation:
    '''Represents interface of a dataset annotation class.

    This class is supposed to be used as a base class for specialized
    implementations designed for particular formats of ground-truth
    data or output of object-detection algorithms.'''

    __slots__ = () # since this class won't have any attribute fields

    def __init__(self, path):
        '''Loads annotation from given path.'''
        raise NotImplementedError('Base DatasetAnnotation.__init__ is used')
    def __getitem__(self, sample_name):
        raise NotImplementedError('Base DatasetAnnotation.__getitem__ is used')
    def __setitem__(self, sample_name, new_value):
        raise NotImplementedError('Base DatasetAnnotation.__setitem__ is used')
    def __delitem__(self, sample_name):
        raise NotImplementedError('Base DatasetAnnotation.__delitem__ is used')
    def __iter__(self):
        raise NotImplementedError('Base DatasetAnnotation.__iter__ is used')

class IVerifier:
    '''A class for actual verifiers to be inhereted from'''
    def __init__(self, AnnotationClass):
        '''Construct a verifier instance with AnnotationClass class

        AnnotationClass class must be the only argument __init__ has. It is here
        to configure verifier properly based on the information which will be
        available in annotation: like bounding-box capatibilities and so on.'''
        pass
    def __call__(self, base_sample, test_sample):
        '''Verify two annotations which belong to gt and tested markup'''
        pass

class IPlugin:
    '''A class for an actual plugin to be inherited from'''
    def inject(self, evaluator):
        derived_classname = self.__class__.__qualname__
        raise NotImplementedError(
            'missing inject() implementation in {}'.format(derived_classname)
        )

class IEvaluationDriver:
    '''A class for actual evaluation drivers to be inhereted from'''
    def __init__(self):
        pass
    def collect(self):
        pass
    def finalize(self):
        pass
