import numpy as np

from graph import Op
from blaze.datashape import coretypes as C
from blaze.datashape.coretypes import promote

from utils import Symbol

#------------------------------------------------------------------------
# Domains
#------------------------------------------------------------------------

Array, Table = Symbol('Array'), Symbol('Table')
one, zero, false, true = xrange(4)

ints      = set([C.int8, C.int16, C.int32, C.int64])
uints     = set([C.uint8, C.uint16, C.uint32, C.uint64])
floats    = set([C.float32, C.float64])
complexes = set([C.complex64, C.complex128])
bools     = set([C.bool_])
string    = set([C.string])

discrete   = ints | uints
continuous = floats | complexes
numeric    = discrete | continuous

array_like, tabular_like = Array, Table
indexable = set([array_like, tabular_like])

universal = set([C.top]) | numeric | indexable | string

#------------------------------------------------------------------------
# Basic Scalar Arithmetic
#------------------------------------------------------------------------

class Add(Op):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = zero
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False

    is_arithmetic = True

class Mul(Op):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = one
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False
    sideffectful = False

    is_arithmetic = True

class Pow(Op):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = zero
    commutative  = False
    associative  = False
    idempotent   = False
    nilpotent    = False
    sideffectful = False

    is_arithmetic = True

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

    is_arithmetic = True
