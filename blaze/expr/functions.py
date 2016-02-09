"""
Dispatches functions like ``sum`` to builtins, numpy, or blaze depending on
input

>>> from blaze import sum, symbol

>>> sum([1, 2, 3])
6

>>> type(sum([1, 2, 3])).__name__
'int'

>>> type(sum(np.array([1, 2, 3], dtype=np.int64))).__name__
'int64'

>>> t = symbol('t', 'var * {x: int, y: int}')
>>> type(sum(t.x)).__name__
'sum'

"""
from __future__ import absolute_import, division, print_function

import math as pymath
from numbers import Number
import numpy as np
import pandas as pd

from multipledispatch import Dispatcher
from ..dispatch import namespace
from ..compatibility import builtins
from . import reductions
from . import math as blazemath
from .core import base
from .expressions import Expr, coalesce


"""
The following code creates reductions equivalent to the following:

@dispatch(Expr)
def sum(expr):
    return blaze.sum(expr)

@dispatch(np.ndarray)
def sum(x):
    return np.sum(x)

@dispatch(object)
def sum(o):
    return builtins.sum(o)


As well as mathematical functions like the following

@dispatch(Expr)
def sqrt(expr):
    return blaze.expr.math.sqrt(expr)

@dispatch(np.ndarray)
def sqrt(x):
    return np.sqrt(x)

@dispatch(object)
def sqrt(o):
    return math.sqrt(o)
"""

math_names = '''abs sqrt sin cos tan sinh cosh tanh acos acosh asin asinh atan atanh
exp log expm1 log10 log1p radians degrees ceil floor trunc isnan'''.split()

binary_math_names = "atan2 copysign fmod hypot ldexp greatest least coalesce".split()

reduction_names = '''any all sum min max mean var std'''.split()

__all__ = math_names + binary_math_names + reduction_names


types = {
    builtins: object,
    np: (np.ndarray, np.number),
    pymath: Number,
    blazemath: Expr,
    reductions: Expr,
}

binary_types = {
    builtins: [(object, object)],
    np: [((np.ndarray, np.number), (np.ndarray, np.number))],
    pymath: [(Number, Number)],
    blazemath: [(Expr, (Expr, base)), (base, Expr)]
}


def _coalesce_objects(lhs, rhs):
    # use pd.isnull for None, NaT, and others
    return lhs if not pd.isnull(lhs) else rhs


def _coalesce_arrays(lhs, rhs):
    np.where(pd.isnull(lhs), rhs, lhs)


fallback_binary_mappings = {
    'greatest': {
        builtins: max,
        np: np.maximum,
        pymath: max,
    },
    'least': {
        builtins: min,
        np: np.minimum,
        pymath: min,
    },
    'coalesce': {
        builtins: _coalesce_objects,
        np: _coalesce_arrays,
        pymath: _coalesce_objects,
        blazemath: coalesce,
    },
}


for funcname in math_names:  # sin, sqrt, ceil, ...
    d = Dispatcher(funcname)

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d


for funcname in binary_math_names:  # hypot, atan2, fmod, ...
    d = Dispatcher(funcname)

    for module, pairs in binary_types.items():
        for pair in pairs:
            if hasattr(module, funcname):
                d.add(pair, getattr(module, funcname))
            elif funcname in fallback_binary_mappings:
                assert module in fallback_binary_mappings[funcname], module.__name__
                d.add(pair, fallback_binary_mappings[funcname][module])

    namespace[funcname] = d
    locals()[funcname] = d


for funcname in reduction_names:  # any, all, sum, max, ...
    d = Dispatcher(funcname)

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d
