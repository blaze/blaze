from ndtable.expr.graph import Op
from operator import itemgetter

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

universal = set([top]) | numeric | indexable

#------------------------------------------------------------------------
# Arity Bases
#------------------------------------------------------------------------

class UnaryOp(Op):
    arity = 1
    abstract = True

class BinaryOp(Op):
    arity = 2
    abstract = True

class NaryOp(Op):
    arity = -1
    abstract = True

#------------------------------------------------------------------------
# Basic Scalar Arithmetic
#------------------------------------------------------------------------

# TODO: Lowercase so they match up with __NAME__ arguments in the
# catalog. Does violate PEP8, rethink.

class add(BinaryOp):
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


class mul(BinaryOp):
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


class transpose(UnaryOp):
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

class abs(UnaryOp):
    # -----------------------
    signature = 'a -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = one
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False
