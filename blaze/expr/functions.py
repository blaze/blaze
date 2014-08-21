"""
Dispatches functions like ``sum`` to builtins, numpy, or blaze depending on
input

>>> sum([1, 2, 3])
6

>>> type(sum([1, 2, 3])).__name__
'int'

>>> type(sum(np.array([1, 2, 3]))).__name__
'int64'

>>> t = TableSymbol('t', '{x: int, y: int}')
>>> type(sum(t.x)).__name__
'sum'

"""


from __future__ import absolute_import, division, print_function

import math
from numbers import Number
import numpy as np

from multipledispatch import Dispatcher
from ..dispatch import dispatch, namespace
from ..compatibility import builtins
from . import scalar
from . import table
from . import Expr, TableExpr, TableSymbol

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

functions = '''sqrt sin cos tan sinh cosh tanh acos acosh asin asinh atan atanh
exp log expm1 log10 log1p radians degrees ceil floor trunc isnan any all sum
min max mean var std'''.split()


__all__ = functions


types = {builtins: object,
         scalar: Expr,
         table: TableExpr,
         np: (np.ndarray, np.number),
         math: Number}


for funcname in functions:
    d = Dispatcher(funcname)
    locals()[funcname] = d
    for module, typ in types.items():
        if hasattr(module, funcname):
            d.add((typ,), getattr(module, funcname))
