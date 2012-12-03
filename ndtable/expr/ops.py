from ndtable.expr.graph import Op

# TODO: bad bad
from ndtable.datashape.coretypes import *

#------------------------------------------------------------------------
# Symbols
#------------------------------------------------------------------------

class Symbol:
    def __init__(self, name):
        self.__name = name
        self.__hash = hash(self.__name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__name == other.__name

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            raise True
        return self.__name != other.__name

    # We require that these be unique in sets and dicts, ergo
    # same hash.
    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return self.__name

#------------------------------------------------------------------------
# Domains
#------------------------------------------------------------------------

Array, Table = Symbol('Array'), Symbol('Table')
one, zero, false, true = xrange(4)

ints      = set([int8, int16, int32, int64])
uints     = set([uint8, uint16, uint32, uint64])
floats    = set([float32, float64])
complexes = set([complex64, complex128])
bools     = set([bool_])
string    = set([string])

discrete   = ints | uints
continuous = floats | complexes
numeric    = discrete | continuous

array_like, tabular_like = Array, Table
indexable = set([array_like, tabular_like])

universal = set([top]) | numeric | indexable | string

#------------------------------------------------------------------------
# Basic Scalar Arithmetic
#------------------------------------------------------------------------

# TODO: Lowercase so they match up with __NAME__ arguments in the
# catalog. Does violate PEP8, rethink.

def arity(op):
    if isinstance(op, Op):
        return op.arity
    else:
        raise ValueError

class ArithmeticOp(Op):
    "Base for unary and binary arithmetic operations"

    is_arithmetic = True

class Add(ArithmeticOp):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [universal, universal]
    # -----------------------

    identity     = zero
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False

class Mul(ArithmeticOp):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [universal, universal]
    # -----------------------

    identity     = one
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False

class Transpose(Op):
    # -----------------------
    arity = 1
    signature = 'a -> a'
    dom = [array_like]
    # -----------------------

    identity     = None
    commutative  = False
    associative  = False
    idempotent   = False
    nilpotent    = True
    sideffectful = False

class Abs(Op):
    # -----------------------
    arity = 1
    signature = 'a -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = None
    commutative  = False
    associative  = False
    idempotent   = True
    nilpotent    = False
    sideffectful = False
