from abc import ABCMeta, abstractmethod

c_types = {
    int: 'int',
    float: 'float',
}

#------------------------------------------------------------------------
# Source Generators
#------------------------------------------------------------------------

class Gen(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def gen(self):
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Gen:
            if hasattr(C, '__str__') and hasattr(C, 'gen'):
                return True
        return NotImplemented
