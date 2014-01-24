"""
Blaze element-wise ufuncs.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['add', 'multiply', 'subtract', 'divide', 'true_divide',
           'floor_divide', 'mod', 'negative',
           'equal', 'not_equal', 'less', 'less_equal', 'greater',
           'greater_equal',
           'logical_or', 'logical_and', 'logical_xor', 'logical_not',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
           'left_shift', 'right_shift',
           'isnan', 'abs', 'log', 'exp', 'logaddexp']

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from ..function import jit_elementwise
from .from_numpy import blazefunc_from_numpy_ufunc
import numpy

@jit_elementwise('a -> a -> a')
def add(a, b):
    return a + b

@jit_elementwise('a -> a -> a')
def multiply(a, b):
    return a * b

@jit_elementwise('a -> a -> a')
def subtract(a, b):
    return a - b

@jit_elementwise('a -> a -> a')
def divide(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def true_divide(a, b):
    return a / b

@jit_elementwise('a -> a -> a')
def floor_divide(a, b):
    return a // b

@jit_elementwise('a -> a -> a')
def mod(a, b):
    return a % b

@jit_elementwise('a -> a')
def negative(a):
    return -a

#------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------

#equal = blazefunc_from_numpy_ufunc(numpy.equal,
#                                       'blaze', 'equal', False)
@jit_elementwise('A..., T -> A..., T -> A..., bool')
def equal(a, b):
    return a == b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def not_equal(a, b):
    return a != b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def less(a, b):
    return a < b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def less_equal(a, b):
    return a <= b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def greater(a, b):
    return a > b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def greater_equal(a, b):
    return a >= b

#------------------------------------------------------------------------
# Logical
#------------------------------------------------------------------------

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_and(a, b):
    return a and b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_or(a, b):
    return a or b

@jit_elementwise('A..., T -> A..., T -> A..., bool')
def logical_xor(a, b):
    return bool(a) ^ bool(b)

@jit_elementwise('A..., T -> A..., bool')
def logical_not(a):
    return not a

#------------------------------------------------------------------------
# Bitwise
#------------------------------------------------------------------------

bitwise_and = blazefunc_from_numpy_ufunc(numpy.bitwise_and,
                                         'blaze', 'bitwise_and', False)

bitwise_or = blazefunc_from_numpy_ufunc(numpy.bitwise_or,
                                         'blaze', 'bitwise_or', False)

bitwise_xor = blazefunc_from_numpy_ufunc(numpy.bitwise_xor,
                                         'blaze', 'bitwise_xor', False)

bitwise_not = blazefunc_from_numpy_ufunc(numpy.bitwise_not,
                                         'blaze', 'bitwise_not', False)

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def left_shift(a, b):
    return a << b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def right_shift(a, b):
    return a >> b

#------------------------------------------------------------------------
# Math
#------------------------------------------------------------------------

@jit_elementwise('A -> A')
def abs(x):
    return builtins.abs(x)

isnan = blazefunc_from_numpy_ufunc(numpy.isnan,
                                       'blaze', 'isnan', False)

log = blazefunc_from_numpy_ufunc(numpy.log,
                                       'blaze', 'log', False)

exp = blazefunc_from_numpy_ufunc(numpy.exp,
                                       'blaze', 'exp', False)

logaddexp = blazefunc_from_numpy_ufunc(numpy.logaddexp,
                                       'blaze', 'logaddexp', False)
