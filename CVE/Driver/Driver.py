__all__ = ('IEvaluationDriver',)

class IEvaluationDriver:
    '''A class for actual evaluation drivers to be inhereted from'''
    def __init__(self):
        pass

    def collect(self):
        pass

    def finalize(self):
        pass
