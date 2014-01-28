"""
Blaze element-wise ufuncs.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['add', 'subtract', 'multiply', 'divide',
           'logaddexp', 'logaddexp2', 'true_divide',
           'floor_divide', 'negative', 'power',
           'remainder', 'mod', 'fmod',
           'absolute', 'abs', 'rint', 'sign',
           'conj', 'real', 'imag',
           'exp', 'exp2', 'log', 'log2', 'log10',
           'sqrt',
           'equal', 'not_equal', 'less', 'less_equal', 'greater',
           'greater_equal',
           'logical_or', 'logical_and', 'logical_xor', 'logical_not',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
           'left_shift', 'right_shift',
           'isnan',
           'mod',

           'degrees', 'radians']

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

from ..function import jit_elementwise
from .from_numpy import blazefunc_from_numpy_ufunc
from .from_dynd import blazefunc_from_dynd_property
import numpy
from dynd import nd, ndt

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

absolute = blazefunc_from_numpy_ufunc(numpy.absolute,
                                       'blaze', 'absolute', False)

abs = absolute

isnan = blazefunc_from_numpy_ufunc(numpy.isnan,
                                       'blaze', 'isnan', False)

power = blazefunc_from_numpy_ufunc(numpy.power,
                                       'blaze', 'power', False)

rint = blazefunc_from_numpy_ufunc(numpy.rint,
                                       'blaze', 'rint', False)

sign = blazefunc_from_numpy_ufunc(numpy.sign,
                                       'blaze', 'sign', False)

degrees = blazefunc_from_numpy_ufunc(numpy.degrees,
                                       'blaze', 'degrees', False)

radians = blazefunc_from_numpy_ufunc(numpy.radians,
                                       'blaze', 'radians', False)

exp = blazefunc_from_numpy_ufunc(numpy.exp,
                                       'blaze', 'exp', False)

exp2 = blazefunc_from_numpy_ufunc(numpy.exp2,
                                       'blaze', 'exp2', False)

log = blazefunc_from_numpy_ufunc(numpy.log,
                                       'blaze', 'log', False)

log2 = blazefunc_from_numpy_ufunc(numpy.log2,
                                       'blaze', 'log2', False)

log10 = blazefunc_from_numpy_ufunc(numpy.log10,
                                       'blaze', 'log10', False)

logaddexp2 = blazefunc_from_numpy_ufunc(numpy.logaddexp2,
                                       'blaze', 'logaddexp2', False)

logaddexp = blazefunc_from_numpy_ufunc(numpy.logaddexp,
                                       'blaze', 'logaddexp', False)

remainder = blazefunc_from_numpy_ufunc(numpy.remainder,
                                       'blaze', 'remainder', False)

mod = blazefunc_from_numpy_ufunc(numpy.mod,
                                       'blaze', 'mod', False)

fmod = blazefunc_from_numpy_ufunc(numpy.fmod,
                                       'blaze', 'fmod', False)

conj = blazefunc_from_numpy_ufunc(numpy.conj,
                                       'blaze', 'conj', False)

sqrt = blazefunc_from_numpy_ufunc(numpy.sqrt,
                                       'blaze', 'sqrt', False)

real = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
            'real', 'blaze', 'real')

imag = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
            'imag', 'blaze', 'imag')
