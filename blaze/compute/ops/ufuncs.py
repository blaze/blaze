"""
Blaze element-wise ufuncs.
"""

from __future__ import absolute_import, division, print_function

ufuncs_from_numpy = [
           'add', 'subtract', 'multiply', 'divide',
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
           'deg2rad', 'rad2deg', 'degrees', 'radians',
           'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
           'invert', 'left_shift', 'right_shift',
           'greater', 'greater_equal', 'less', 'less_equal',
           'not_equal', 'equal',
           'logical_and', 'logical_or', 'logical_xor', 'logical_not',
           'maximum', 'minimum', 'fmax', 'fmin',
           'isfinite', 'isinf', 'isnan',
           'signbit', 'copysign', 'nextafter', 'ldexp',
           'fmod', 'floor', 'ceil', 'trunc']

__all__ = ufuncs_from_numpy + ['real', 'imag']

try:
    import __builtin__ as builtins
except ImportError:
    import builtins

import numpy
from dynd import ndt

from .from_numpy import blazefunc_from_numpy_ufunc
from .from_dynd import blazefunc_from_dynd_property

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
