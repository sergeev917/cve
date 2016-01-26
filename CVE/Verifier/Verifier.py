__all__ = ('IVerifier',)

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
