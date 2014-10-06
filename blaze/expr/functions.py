"""
Dispatches functions like ``sum`` to builtins, numpy, or blaze depending on
input

>>> sum([1, 2, 3])
6

>>> type(sum([1, 2, 3])).__name__
'int'

>>> type(sum(np.array([1, 2, 3], dtype=np.int64))).__name__
'int64'

>>> t = Symbol('t', '{x: int, y: int}')
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
from . import table
from . import math as blazemath
from .expressions import Expr, Symbol
from .broadcast import broadcast


"""
The following code creates functions equivalent to the following:

@dispatch(Expr)
def sum(expr):
    return scalar.sum(expr)

@dispatch(TableExpr)
def sum(expr):
    return table.sum(expr)

@dispatch(np.ndarray)
def sum(x):
    return np.sum(x)

@dispatch(object)
def sum(o):
    return builtins.sum(o)
"""

math_functions = '''sqrt sin cos tan sinh cosh tanh acos acosh asin asinh atan atanh
exp log expm1 log10 log1p radians degrees ceil floor trunc isnan'''.split()

reductions = '''any all sum min max mean var std'''.split()

__all__ = math_functions + reductions


types = {builtins: object,
         np: (np.ndarray, np.number),
         pymath: Number}


def switch(funcname, x):
    f = getattr(blazemath, funcname)
    if iscollection(x.dshape):
        return broadcast(f, x)
    else:
        return f(x)


for funcname in math_functions:  # sin, sqrt, ceil, ...
    d = Dispatcher(funcname)

    d.add((Expr,), curry(switch, funcname))

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d


for funcname in reductions:  # any, all, sum, max, ...
    d = Dispatcher(funcname)

    d.add((Expr,), getattr(table, funcname))

    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))

    namespace[funcname] = d
    locals()[funcname] = d
