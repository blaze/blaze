"""
Create signatures wrapped around the umath functions.
"""

import numpy as np
from pprint import pprint
from types import FunctionType
#from blaze.datashape import from_numpy

#------------------------------------------------------------------------
# Sorts
#------------------------------------------------------------------------

MAP     = 0
ZIPWITH = 1
SCAN    = 2
REDUCE  = 3
OUTER   = 4

class NumpySort(object):
    """
    """

    def __init__(self, method, fn):
        self.method = method
        self.fn = fn

    @property
    def map(self):
        return self.method == MAP

    @property
    def zipwith(self):
        return self.method == ZIPWITH

    @property
    def scan(self):
        return self.method == SCAN

    @property
    def reduce(self):
        return self.method == REDUCE

    @property
    def outer(self):
        return self.method == OUTER

    def __repr__(self):
        return repr(self.fn)

class NumpyOp(object):
    """
    """

    def __init__(self, fn, pyfn=None, symbol=None, name=None,
            nin=None, nout=None):

        self.fn = fn
        self.pyfn = pyfn
        self.symbol = symbol
        self.name = name or fn.__name__
        self.sig = {}

        # Map arity
        if nin:
            self.nin = nin
        elif hasattr(fn, 'nin'):
            self.nin = fn.nin
        else:
            assert isinstance(fn, FunctionType)
            self.nin = fn.func_code.co_argcount

        # Map arity
        if nout:
            self.nout = nout
        elif hasattr(fn, 'nout'):
            self.nout = fn.nout
        else:
            self.nout = 1

        self._map_signature()

    # should only be called by constructor
    def _map_signature(self):
        if hasattr(self.fn, 'types'):
            for signature in self.fn.types:
                dom, cod = signature.split('->')

                domi = tuple(map(np.dtype, dom))
                codi = tuple(map(np.dtype, cod))

                self.sig[domi] = codi
        else:
            raise NotImplementedError

    def __eq__(self, other):
        return self.fn == other.fn

    def __repr__(self):
        return "np.%s" % self.name

#------------------------------------------------------------------------
# Defs
#------------------------------------------------------------------------

add      = NumpyOp(np.add, 'Add', '+')
multiply = NumpyOp(np.multiply, 'Mult', '*')
subtract = NumpyOp(np.subtract, 'Sub', '-')
divide   = NumpyOp(np.divide, 'Div', '/')
mod      = NumpyOp(np.mod, 'Mod', '%')
power    = NumpyOp(np.power, 'Pow', '**')

bit_not = NumpyOp(np.bitwise_not, 'Invert', '!')
bit_and = NumpyOp(np.bitwise_and, 'BitAnd', '&')
bit_or  = NumpyOp(np.bitwise_or, 'BitOr', '|')
bit_xor = NumpyOp(np.bitwise_xor, 'BitXor', '^')

logic_and = NumpyOp(np.logical_and, "LogicAnd")
logic_not = NumpyOp(np.logical_not, "LogicNot")
logic_or  = NumpyOp(np.logical_or, "LogicOr")

eq  = NumpyOp(np.equal, 'Eq', '==')
neq = NumpyOp(np.not_equal, 'Neq', '!=')

gt  = NumpyOp(np.greater, 'Gt', '>')
gte = NumpyOp(np.greater_equal, 'Gte', '>=')
lt  = NumpyOp(np.less, 'Lt', '<')
lte = NumpyOp(np.less_equal, 'Lte', '<=')

sqrt  = NumpyOp(np.sqrt, 'Sqrt')
exp   = NumpyOp(np.exp, 'Exp')
log   = NumpyOp(np.log, 'Log')
log2  = NumpyOp(np.log2, 'Log2')
log10 = NumpyOp(np.log10, 'Log10')
sin   = NumpyOp(np.sin, 'Sin')
cos   = NumpyOp(np.cos, 'Cos')
tan   = NumpyOp(np.tan, 'Tan')
sinh  = NumpyOp(np.sinh, 'Sinh')
cosh  = NumpyOp(np.cosh, 'Cosh')
tanh  = NumpyOp(np.tanh, 'Tanh')

# For Python calling the application strategy is stored on the
# operator itself so rather unnaturally we have the combinator
# reach into the operator to clal itself.

def MkMap(op):
    return NumpySort(MAP, op.fn)

def MkZipWith(op):
    return NumpySort(ZIPWITH, op.fn)

def MkReduce(op):
    return NumpySort(REDUCE, op.fn.reduce)

def MkScan(op):
    return NumpySort(SCAN, op.fn.accumulate)

def MkOuter(op):
    return NumpySort(OUTER, op.fn.outer)

#------------------------------------------------------------------------
# Ops
#------------------------------------------------------------------------

ops =  [
      'add'
    , 'multiply'
    , 'subtract'
    , 'divide'
    , 'mod'
    , 'power'

    , 'bit_not'
    , 'bit_and'
    , 'bit_or'
    , 'bit_xor'

    , 'logic_and'
    , 'logic_not'
    , 'logic_or'

    , 'eq'
    , 'neq'

    , 'gt'
    , 'gte'
    , 'lt'
    , 'lte'

    , 'sqrt'
    , 'exp'
    , 'log'
    , 'sqrt'
    , 'log10'
    , 'log2'
    , 'cos'
    , 'cosh'
    , 'sin'
    , 'sinh'
    , 'tan'
    , 'tanh'
]

sorts = {}

for op in ops:
    op = locals()[op]
    if op.nin == 1:
        sorts['Map_%s' % op.pyfn] = MkMap(op)
    elif op.nin == 2:
        sorts['ZipWith_%s' % op.pyfn] = MkZipWith(op)
        sorts['Reduce_%s' % op.pyfn]  = MkReduce(op)
        sorts['Scan_%s' % op.pyfn]    = MkScan(op)
        sorts['Outer_%s' % op.pyfn]   = MkOuter(op)

#------------------------------------------------------------------------
# Module Interface
#------------------------------------------------------------------------

constructors = {
    'array': np.ndarray,
    'slice': slice,
}

# Types are always tuples, unit types are 1-tuples
# Parameterized types are 2-tuples with a the first element being
# a list of tuples of the form (name, ty) with ty string
# reference to other type.
types = {
    'int'   : (np.int,),
    'float' : (np.float),
    'bool'  : (np.bool,),
    'Array' : (np.ndarray, [('T',)]),
}
