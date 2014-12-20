from __future__ import absolute_import, division, print_function

try:
    from dynd.nd import array
except ImportError:
    array = type(None)

from ..expr import *
from .core import compute
from ..dispatch import dispatch

__all__ = ()


@dispatch(Slice, array)
def compute_up(expr, data, **kwargs):
    return data[expr.index]


@dispatch(Field, array)
def compute_up(expr, data, **kwargs):
    return getattr(data, expr._name)


@dispatch(Broadcast, array)
def compute_up(t, x, **kwargs):
    d = dict((t._child[c]._expr, getattr(x, c)) for c in t._child.fields)
    return compute(t._expr, d)


@dispatch(BinOp, array, array)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, array)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)


@dispatch(Not, array)
def compute_up(t, x, **kwargs):
    return ~x


@dispatch(USub, array)
def compute_up(t, x, **kwargs):
    return 0-x
