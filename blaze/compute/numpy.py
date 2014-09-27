from __future__ import absolute_import, division, print_function

import numpy as np

from blaze.expr import Reduction, Column, Projection, ColumnWise, Selection
from blaze.expr import Distinct, Sort, Head, Label, ReLabel, Union, Expr
from blaze.expr import std, var, count, nunique
from blaze.expr.scalar import BinOp, UnaryOp, USub, Not

from .core import base, compute
from ..dispatch import dispatch
from blaze.api.into import into
from pandas import DataFrame, Series

__all__ = ['np']


@dispatch(Column, np.ndarray)
def compute_up(c, x, **kwargs):
    if x.dtype.names and c._name in x.dtype.names:
        return x[c._name]
    if not x.dtype.names and x.shape[1] == len(c.child.names):
        return x[:, c.child.names.index(c._name)]
    raise NotImplementedError()


@dispatch(Projection, np.ndarray)
def compute_up(t, x, **kwargs):
    if all(col in x.dtype.names for col in t.names):
        return x[t.names]
    if not x.dtype.names and x.shape[1] == len(t.child.names):
        return x[:, [t.child.names.index(col) for col in t.names]]
    raise NotImplementedError()


@dispatch(ColumnWise, np.ndarray)
def compute_up(t, x, **kwargs):
    d = dict((t.child[c].expr, x[c]) for c in t.child.names)
    return compute(t.expr, d)


@dispatch(BinOp, np.ndarray, (np.ndarray, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, base, np.ndarray)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, np.ndarray)
def compute_up(t, x, **kwargs):
    return getattr(np, t.symbol)(x)


@dispatch(Not, np.ndarray)
def compute_up(t, x, **kwargs):
    return ~x


@dispatch(USub, np.ndarray)
def compute_up(t, x, **kwargs):
    return -x


@dispatch(Selection, np.ndarray)
def compute_up(t, x, **kwargs):
    predicate = compute(t.predicate, {t.child: x})
    return x[predicate]


@dispatch(count, np.ndarray)
def compute_up(t, x, **kwargs):
    return len(x)


@dispatch(nunique, np.ndarray)
def compute_up(t, x, **kwargs):
    return len(np.unique(x))


@dispatch(Reduction, np.ndarray)
def compute_up(t, x, **kwargs):
    return getattr(x, t.symbol)()


@dispatch((std, var), np.ndarray)
def compute_up(t, x, **kwargs):
    return getattr(x, t.symbol)(ddof=t.unbiased)


@dispatch(Distinct, np.ndarray)
def compute_up(t, x, **kwargs):
    return np.unique(x)


@dispatch(Sort, np.ndarray)
def compute_up(t, x, **kwargs):
    if (t.key in x.dtype.names or
        isinstance(t.key, list) and all(k in x.dtype.names for k in t.key)):
        result = np.sort(x, order=t.key)
    elif t.key:
        raise NotImplementedError("Sort key %s not supported" % str(t.key))
    else:
        result = np.sort(x)

    if not t.ascending:
        result = result[::-1]

    return result


@dispatch(Head, np.ndarray)
def compute_up(t, x, **kwargs):
    return x[:t.n]


@dispatch(Label, np.ndarray)
def compute_up(t, x, **kwargs):
    return np.array(x, dtype=[(t.label, x.dtype.type)])


@dispatch(ReLabel, np.ndarray)
def compute_up(t, x, **kwargs):
    types = [x.dtype[i] for i in range(len(x.dtype))]
    return np.array(x, dtype=list(zip(t.names, types)))


@dispatch(Selection, np.ndarray)
def compute_up(sel, x, **kwargs):
    return x[compute(sel.predicate, {sel.child: x})]


@dispatch(Union, np.ndarray, tuple)
def compute_up(expr, example, children, **kwargs):
    return np.concatenate(list(children), axis=0)



@dispatch(Expr, np.ndarray)
def compute_up(t, x, **kwargs):
    if x.ndim > 1 or isinstance(x, np.recarray) or x.dtype.fields is not None:
        df = DataFrame(columns=t.child.names)
    else:
        df = Series(name=t.child.names[0])
    return compute_up(t, into(df, x), **kwargs)


@dispatch(np.ndarray)
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize
