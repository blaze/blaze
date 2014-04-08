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

ufuncs_from_dynd = ['real', 'imag']

reduction_ufuncs = ['any', 'all', 'sum', 'product', 'min', 'max']

__all__ = ufuncs_from_numpy + ufuncs_from_dynd + reduction_ufuncs

import numpy as np
from dynd import ndt, _lowlevel

from .from_numpy import blazefunc_from_numpy_ufunc
from .from_dynd import blazefunc_from_dynd_property
from ..function import ReductionBlazeFunc

#------------------------------------------------------------------------
# UFuncs converted from NumPy
#------------------------------------------------------------------------

for name in ufuncs_from_numpy:
    globals()[name] = blazefunc_from_numpy_ufunc(getattr(np, name),
                                                 'blaze', name, False)

#------------------------------------------------------------------------
# UFuncs from DyND
#------------------------------------------------------------------------

real = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
                                    'real', 'blaze', 'real')

imag = blazefunc_from_dynd_property([ndt.complex_float32, ndt.complex_float64],
                                    'imag', 'blaze', 'imag')

#------------------------------------------------------------------------
# Reduction UFuncs from NumPy
#------------------------------------------------------------------------

bools = np.bool,
ints = np.int8, np.int16, np.int32, np.int64,
floats = np.float32, np.float64
complexes = np.complex64, np.complex128,

reductions = {('any', np.logical_or,    False, bools),
              ('all', np.logical_and,  True, bools),
              ('sum', np.add,          0, ints + floats + complexes),
              ('product', np.multiply, 1, ints + floats + complexes),
              ('min', np.minimum,      None, bools + ints + floats + complexes),
              ('max', np.maximum,      None, bools + ints + floats + complexes)}

for name, np_op, ident, types in reductions:
    x = ReductionBlazeFunc('blaze', name)
    for typ in types:
        x.add_overload('(%s) -> %s' % (typ.__name__, typ.__name__),
                 _lowlevel.ckernel_deferred_from_ufunc(np_op,
                                                       (typ,) * 3,
                                                       False),
                 associative=True, commutative=True,
                 identity=ident)
        locals()[name] = x
