import numpy as np
from types import FunctionType

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
    self.sig = None

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
        self.sig = {}
        for signature in self.fn.types:
            dom, cod = signature.split('->')
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

arith_add      = AArith(np.add, 'Add', '+')
arith_multiply = AArith(np.multiply, 'Mult', '*')
arith_subtract = AArith(np.subtract, 'Sub', '-')
arith_divide   = AArith(np.divide, 'Div', '/')
arith_mod      = AArith(np.mod, 'Mod', '%')
arith_power    = AArith(np.power, 'Pow', '**')

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
sqrt  = ATransc(np.sqrt)
exp   = ATransc(np.exp)
log   = ATransc(np.log)
log2  = ATransc(np.log2)
log10 = ATransc(np.log10)
cos   = ATransc(np.cos)
cosh  = ATransc(np.cosh)
sin   = ATransc(np.sin)
sinh  = ATransc(np.sinh)
tan   = ATransc(np.tan)
tanh  = ATransc(np.tanh)

prim_types = [ATransc, ABit, ALogic, AArith, ATransc]

__all__ =  [
      'arith_add'
    , 'arith_multiply'
    , 'arith_subtract'
    , 'arith_divide'
    , 'arith_mod'
    , 'arith_power'

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
]
