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

    @staticmethod
    def register(name, cls):
        assert name not in Type.registry
        Type.registry[name] = cls

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

    def __eq__(self, other):
        if type(other) is DataShape:
            return self.operands == other.operands
        else:
            return False

    def __repr__(self):
        return str(self)

class CType(DataShape):

    def __init__(self, ctype):
        self.parameters = [ctype]
        self.name = ctype
        Type.register(ctype, self)

    def __str__(self):
        return str(self.parameters[0])

    def __eq__(self, other):
        if type(other) is CType:
            return self.parameters[0] == other.parameters[0]
        else:
            return False

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

    def __eq__(self, other):
        if type(other) is Integer:
            return self.val == other.val
        else:
            return False

    def __str__(self):
        return str(self.val)

class TypeVar(Term):

    def __init__(self, symbol):
        self.symbol = symbol

    def free(self):
        return set([self.symbol])

    def __str__(self):
        return str(self.symbol)

    def __eq__(self, other):
        if isinstance(other, TypeVar):
            return self.symbol == other.symbol
        else:
            return False

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

class Stream(Term):

    def __init__(self, ret_type):
        self.parameters = [ret_type]

    def __str__(self):
        return 'Stream '+ repr(self.parameters[0])

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

_int       = CType('int')
_float     = CType('float')
_long      = CType('long')
_bool      = CType('bool')
double     = CType('double')
short      = CType('short')
longdouble = CType('longdbouble')
char       = CType('char')

uint       = CType('uint')
ulong      = CType('ulong')
ulonglong  = CType('ulonglong')

int8       = CType('int8')
int16      = CType('int16')
int32      = CType('int32')
int64      = CType('int64')

uint8      = CType('uint8')
uint16     = CType('uint16')
uint32     = CType('uint32')
uint64     = CType('uint64')

complex64  = CType('complex64')
complex128 = CType('complex128')
complex256 = CType('complex256')

void       = CType('void')
pyobj      = CType('PyObject')
