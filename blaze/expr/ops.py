import numpy as np

from graph import Op
from blaze.datashape import coretypes as C

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

def arity(op):
    if isinstance(op, Op):
        return op.arity
    else:
        raise ValueError

def promote(*operands):
    "Compute the promoted datashape, uses NumPy..."
    dtypes = [C.to_dtype(op.datashape) for op in operands]
    result_dtype = reduce(np.promote_types, dtypes)
    datashape = C.CType.from_dtype(result_dtype)
    return datashape

class ArithmeticOp(Op):
    "Base for unary and binary arithmetic operations"

    is_arithmetic = True

class BinaryArithmeticOp(ArithmeticOp):

    def __init__(self, op, operands):
        super(BinaryArithmeticOp, self).__init__(op, operands)
        self.datashape = promote(*operands)

class Add(BinaryArithmeticOp):
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

class Mul(BinaryArithmeticOp):
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

class Pow(Op):
    # -----------------------
    arity = 2
    signature = 'a -> a -> a'
    dom = [universal, numeric]
    # -----------------------

    identity     = zero
    commutative  = False
    associative  = False
    idempotent   = False
    nilpotent    = False
    sideffectful = False

    def __init__(self, op, operands):
        super(Pow, self).__init__(op, operands)
        self.datashape = promote(*operands)

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

    def __init__(self, op, operands):
        super(Abs, self).__init__(op, operands)
        self.datashape = operands[0].datashape


