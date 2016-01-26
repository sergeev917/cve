__all__ = ('IPlugin',)

class IPlugin:
    '''A class for actual plugins to be inhereted from'''
    def __init__(self):
        pass

    def inject(self, evaluator):
        pass
