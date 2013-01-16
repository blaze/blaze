from graph import Op
from blaze.datashape import coretypes as C

from utils import Symbol

#------------------------------------------------------------------------
# Domain Constraints
#------------------------------------------------------------------------

# TODO: Dependency on graph is not desirable
Array, Table = Symbol('Array'), Symbol('Table')
one, zero, false, true = xrange(4)

ints      = set([C.int8, C.int16, C.int32, C.int64])
uints     = set([C.uint8, C.uint16, C.uint32, C.uint64])
floats    = set([C.float32, C.float64])
complexes = set([C.complex64, C.complex128])
bools     = set([C.bool_])
string    = set([C.String, C.Varchar, C.Blob])

discrete   = ints | uints
reals      = ints | floats
continuous = floats | complexes
numeric    = discrete | continuous

array_like, table_like = Array, Table
indexable = set([array_like, table_like])

universal = set([C.top]) | numeric | indexable | string

#------------------------------------------------------------------------
# Basic Scalar Arithmetic
#------------------------------------------------------------------------

# These are abstract graph nodes for Operations.

# TODO: remove this in favor of new module system

class Add(Op):
    # -----------------------
    arity = 2
    signature = '(a,a) -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = zero
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False

class Mul(Op):
    # -----------------------
    arity = 2
    signature = '(a,a) -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = one
    commutative  = True
    associative  = True
    idempotent   = False
    nilpotent    = False

class Pow(Op):
    # -----------------------
    arity = 2
    signature = '(a,b) -> a'
    dom = [numeric, numeric]
    # -----------------------

    identity     = zero
    commutative  = False
    associative  = False
    idempotent   = False
    nilpotent    = False

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
