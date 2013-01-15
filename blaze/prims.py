"""
Create signatures wrapped around the umath functions.
"""

import numpy as np
from types import FunctionType
from blaze.datashape import from_numpy

#------------------------------------------------------------------------
# Lifted Primitives
#------------------------------------------------------------------------

class Prim(object):
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
        return "Prim(%s)" % self.name

#------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------

class ATransc(Prim):
  pass

class AArith(Prim):
  pass

class ALogic(Prim):
  pass

class ABit(Prim):
  pass

class ACmp(Prim):
  pass

#------------------------------------------------------------------------
# Defs
#------------------------------------------------------------------------

add      = AArith(np.add, 'Add', '+')
multiply = AArith(np.multiply, 'Mult', '*')
subtract = AArith(np.subtract, 'Sub', '-')
divide   = AArith(np.divide, 'Div', '/')
mod      = AArith(np.mod, 'Mod', '%')
power    = AArith(np.power, 'Pow', '**')

bit_not = ABit(np.bitwise_not, 'Invert', '!')
bit_and = ABit(np.bitwise_and, 'BitAnd', '&')
bit_or  = ABit(np.bitwise_or, 'BitOr', '|')
bit_xor = ABit(np.bitwise_xor, 'BitXor', '^')

logic_and = ALogic(np.logical_and, "And")
logic_not = ALogic(np.logical_not, "Not")
logic_or  = ALogic(np.logical_or, "Or")

cmp_eq  = ACmp(np.equal, 'Eq', '==')
cmp_neq = ACmp(np.not_equal, 'Neq', '!=')

cmp_gt  = ACmp(np.greater, 'Gt', '>')
cmp_gte = ACmp(np.greater_equal, 'Gte', '>=')
cmp_lt  = ACmp(np.less, 'Lt', '<')
cmp_lte = ACmp(np.less_equal, 'Lte', '<=')

sqrt  = ATransc(np.sqrt)
exp   = ATransc(np.exp)
log   = ATransc(np.log)
log2  = ATransc(np.log2)
log10 = ATransc(np.log10)
sin   = ATransc(np.sin)
cos   = ATransc(np.cos)
tan   = ATransc(np.tan)
sinh  = ATransc(np.sinh)
cosh  = ATransc(np.cosh)
tanh  = ATransc(np.tanh)

prim_types = [ATransc, ABit, ALogic, AArith, ATransc]

__all__ =  [
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

    , 'cmp_eq'
    , 'cmp_neq'

    , 'cmp_gt'
    , 'cmp_gte'
    , 'cmp_lt'
    , 'cmp_lte'

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

      'ATransc'
    , 'ABit'
    , 'ALogic'
    , 'AArith'
    , 'ATransc'
]
