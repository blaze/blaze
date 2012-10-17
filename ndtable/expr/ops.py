from ndtable.expr.graph import Op

# TODO: bad bad
from ndtable.datashape.coretypes import *

#------------------------------------------------------------------------
# Domains
#------------------------------------------------------------------------

one, zero, false, true = xrange(4)

bools     = set([bool_])
ints      = set([int8, int16, int32, int64])
uints     = set([uint8, uint16, uint32, uint64])
floats    = set([float32, float64])
complexes = set([complex64, complex128])

discrete   = ints | uints
continuous = floats | complexes
numeric    = discrete | continuous

array_like, tabular_like = xrange(2)
indexable = set([array_like, tabular_like])

universal = set([top])

#------------------------------------------------------------------------
# Arity Bases
#------------------------------------------------------------------------

class UnaryOp(Op):
    arity = 1

class BinaryOp(Op):
    arity = 2

class NaryOp(Op):
    arity = -1

#------------------------------------------------------------------------
# Basic Scalar Arithmetic
#------------------------------------------------------------------------

class Add(BinaryOp):
    # -----------------------
    signature = 'a -> a -> a'
    dom = [universal, universal]
    # -----------------------

    identity     = zero
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False


class Mul(BinaryOp):
    # -----------------------
    signature = 'a -> a -> a'
    dom = [universal, universal]
    # -----------------------

    identity     = one
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False


class Transpose(UnaryOp):
    # -----------------------
    signature = 'a -> a'
    dom = [universal, universal]
    # -----------------------

    identity     = None
    commutative  = False
    associative  = False
    idempotent   = False
    nilpotent    = True
    sideffectful = False

#------------------------------------------------------------------------
# Catelog
#------------------------------------------------------------------------

functions = {
    'add' : Add,
    'mul' : Mul,
}
