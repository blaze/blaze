from __future__ import absolute_import, division, print_function

from dynd import nd

from ..expr import *
from .core import base, compute
from ..dispatch import dispatch
from ..api.into import into

__all__ = 'nd',


@dispatch(Slice, nd.array)
def compute_up(expr, data, **kwargs):
    return data[expr.index]


@dispatch(Field, nd.array)
def compute_up(expr, data, **kwargs):
    return getattr(data, expr._name)


@dispatch(Broadcast, nd.array)
def compute_up(t, x, **kwargs):
    d = dict((t._child[c]._expr, getattr(x, c)) for c in t._child.fields)
    return compute(t._expr, d)


@dispatch(BinOp, nd.array, (nd.array, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, base, nd.array)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(Not, nd.array)
def compute_up(t, x, **kwargs):
    return ~x


@dispatch(USub, nd.array)
def compute_up(t, x, **kwargs):
    return 0-x
