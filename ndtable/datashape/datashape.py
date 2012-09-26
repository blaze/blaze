"""
This defines the DataShape "type system".

data CType = int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | ...
data Size = Integer | Variable | Function | Stream | Var n
data Type = Size | CType | Record
data DataShape = Size : Type
"""

import ctypes
from numbers import Integral
from operator import methodcaller
from collections import Mapping

free_vars = methodcaller('free')

def shape_coerce(ob):
    if type(ob) is int:
        return Integer(ob)
    else:
        raise NotImplementedError()

def flatten(it):
    for a in it:
        if a.composite:
            for b in iter(a):
                yield b
        else:
            yield a

class Type(type):
    registry = {}
    def __new__(meta, name, bases, dct):
        cls = type(name, bases, dct)
        Type.registry[name] = cls
        return cls

class DataShape(object):
    __metaclass__ = Type

    composite = False
    name = False

    def __init__(self, operands=None, name=False):

        if type(operands) is DataShape:
            self.operands = operands

        elif len(operands) > 0:
            self.operands = tuple(flatten(operands))
            self.composite = True
        else:
            self.operands = tuple()
            self.composite = False

        if name:
            self.name = name
            self.__metaclass__.registry[name] = self

        #if isinstance(operands, Iterable):
            #itx = reduce(compose, operands)
        #else:
            #self.operands = operands

    def __getitem__(self, index):
        return self.operands[index]

    def __getslice__(self, start, stop):
        return self.operands[start:stop]

    def __rmul__(self, other):
        if not isinstance(other, DataShape):
            other = shape_coerce(other)
        return compose(other, self)

    def __mul__(self, other):
        if not isinstance(other, DataShape):
            other = shape_coerce(other)
        return compose(other, self)

    def __str__(self):
        if self.name:
            return self.name
        else:
            return ' '.join(map(str, self.operands))

    def __repr__(self):
        return str(self)

    @classmethod
    def register(cls, name, ds):
        cls.__metaclass__.registry[name] = ds

    @property
    def defined(self):
        return  self.__metaclass__.registry

class CType(DataShape):

    def __init__(self, ctype):
        self.parameters = [ctype]
        self.name = ctype

    def __str__(self):
        return str(self.parameters[0])

class Term(DataShape):
    abstract = True

    # Type constructor
    def __init__(self, *parameters):
        self.parameters = parameters

    def __repr__(self):
        return str(self)

class Integer(Term):

    def __init__(self, i):
        assert isinstance(i, Integral)
        self.val = i

    def free(self):
        return set([])

    def __str__(self):
        return str(self.val)

class TypeVar(Term):

    def __init__(self, symbol):
        self.symbol = symbol

    def free(self):
        return set([self.symbol])

    def __str__(self):
        return str(self.symbol)

class Enum(Term):

    def __str__(self):
        return 'Enum (' + ','.join(map(str,self.parameters)) + ')'

class Tuple(Term):

    def __getitem__(self, index):
        return self.parameters[index]

    def __getslice__(self, start, stop):
        return self.operands[start:stop]

    def __str__(self):
        return '(' + ','.join(map(str,self.parameters)) + ')'

class Function(Term):

    # Same as Numba notation
    def __init__(self, arg_type, ret_type):
        self.arg_type = arg_type
        self.ret_type = ret_type

        self.parameters = [arg_type, ret_type]

    def free(self):
        return map(free_vars, self.arg_type) | free_vars(self.ret_type)

    def __str__(self):
        return str(self.arg_type) + " -> " + str(self.ret_type)

class Record(DataShape, Mapping):

    def __init__(self, **kwargs):
        self.d = kwargs
        self.k = kwargs.keys()
        self.v = kwargs.values()

    def free(self):
        return set(self.k)

    def __call__(self, key):
        return self.d[key]

    def __iter__(self):
        return zip(self.k, self.v)

    def __len__(self):
        return len(self.k)

    def __str__(self):
        return '( ' + ''.join([('%s = %s, ' % (k,v)) for k,v in zip(self.k, self.v)]) + ')'

    def __repr__(self):
        return 'Record ' + repr(self.d)

def compose(A,B):
    """
    Datashape composition operator.
    """
    if A.composite and B.composite:
        f = A.operands
        g = B.operands

    elif A.composite:
        f = A.operands
        g = (B,)

    elif B.composite:
        f = (A,)
        g = B.operands

    else:
        f = (A,)
        g = (B,)

    return DataShape(operands=(f+g))

int64 = CType('int64')
int32 = CType('int32')
int16 = CType('int16')
int8  = CType('int8')

if __name__ == '__main__':

    w = TypeVar('w')
    x = TypeVar('x')
    y = TypeVar('y')
    z = TypeVar('z')

    Quaternion = operands=(z*y*x*w)
    RGBA = Record(R=int16, G=int16, B=int16, A=int8)

    RGBA * 800 * 600

    T = Tuple(1,2,3)

    DataShape.register('Quaternion', Quaternion)
    DataShape.register('RGBA', RGBA)

    print(T)
    print(Quaternion * 800 * 600)
