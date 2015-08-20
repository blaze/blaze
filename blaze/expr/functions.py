"""
Dispatches functions like ``sum`` to builtins, numpy, or blaze depending on
input

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
from datashape.predicates import iscollection, isscalar
from toolz import curry

from multipledispatch import Dispatcher
from ..dispatch import dispatch, namespace
from ..compatibility import builtins
from . import reductions
from . import math as blazemath
from .expressions import Expr, symbol


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

reduction_names = '''any all sum min max mean var std'''.split()

__all__ = math_names + reduction_names


types = {builtins: object,
         np: (np.ndarray, np.number),
         pymath: Number,
         blazemath: Expr,
         reductions: Expr}


for funcname in math_names:  # sin, sqrt, ceil, ...
    d = Dispatcher(funcname)

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d


for funcname in reduction_names:  # any, all, sum, max, ...
    d = Dispatcher(funcname)

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d
