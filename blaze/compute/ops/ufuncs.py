"""
Blaze element-wise ufuncs.
"""

from __future__ import absolute_import, division, print_function

ufuncs_from_numpy = [
           'logaddexp', 'logaddexp2', 'true_divide',
           'floor_divide', 'negative', 'power',
           'remainder', 'mod', 'fmod',
           'absolute', 'abs', 'rint', 'sign',
           'conj',
           'exp', 'exp2', 'log', 'log2', 'log10', 'expm1', 'log1p',
           'sqrt', 'square', 'reciprocal',
           'sin', 'cos', 'tan', 'arcsin',
           'arccos', 'arctan', 'arctan2',
           'hypot', 'sinh', 'cosh', 'tanh',
           'arcsinh', 'arccosh', 'arctanh',
           'deg2rad', 'rad2deg',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
           'isnan',
           'degrees', 'radians',
           'maximum', 'minimum', 'fmax', 'fmin']

__all__ = ufuncs_from_numpy + \
          ['add', 'subtract', 'multiply', 'divide',
           'real', 'imag',
           'equal', 'not_equal', 'less', 'less_equal', 'greater',
           'greater_equal',
           'logical_or', 'logical_and', 'logical_xor', 'logical_not',
           'left_shift', 'right_shift',
           'mod']

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

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def left_shift(a, b):
    return a << b

@jit_elementwise('A..., T : integral -> A..., T -> A..., T')
def right_shift(a, b):
    return a >> b

#------------------------------------------------------------------------
# UFuncs converted from NumPy
#------------------------------------------------------------------------

for name in ufuncs_from_numpy:
    globals()[name] = blazefunc_from_numpy_ufunc(getattr(numpy, name),
                                                 'blaze', name, False)

#------------------------------------------------------------------------
# UFuncs from DyND
#------------------------------------------------------------------------

real = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
            'real', 'blaze', 'real')

imag = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
            'imag', 'blaze', 'imag')
