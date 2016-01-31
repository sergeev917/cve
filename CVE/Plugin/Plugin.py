__all__ = ('IPlugin',)

class IPlugin:
    '''A class for an actual plugin to be inherited from'''
    def inject(self, evaluator):
        derived_classname = self.__class__.__qualname__
        raise NotImplementedError(
            'missing inject() implementation in {}'.format(derived_classname)
        )
