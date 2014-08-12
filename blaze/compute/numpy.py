from __future__ import absolute_import, division, print_function

import numpy as np
from blaze.expr.table import *
from blaze.expr.scalar import BinOp, UnaryOp, USub
from datashape import Record
from .core import base, compute
from ..dispatch import dispatch

__all__ = ['compute_one', 'np', 'chunks']


@dispatch(Column, np.ndarray)
def compute_one(c, x, **kwargs):
    if x.dtype.names and c.column in x.dtype.names:
        return x[c.column]
    if not x.dtype.names and x.shape[1] == len(c.child.columns):
        return x[:, c.child.columns.index(c.column)]
    raise NotImplementedError()


@dispatch(Projection, np.ndarray)
def compute_one(t, x, **kwargs):
    if all(col in x.dtype.names for col in t.columns):
        return x[t.columns]
    if not x.dtype.names and x.shape[1] == len(c.child.columns):
        return x[:, [c.child.columns.index(col) for col in c.columns]]
    raise NotImplementedError()


@dispatch(ColumnWise, np.ndarray)
def compute_one(t, x, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c].scalar_symbol, x[c]) for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(BinOp, np.ndarray, (np.ndarray, base))
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, base, np.ndarray)
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, np.ndarray)
def compute_one(t, x, **kwargs):
    return getattr(np, t.symbol)(x)

@dispatch(Not, np.ndarray)
def compute_one(t, x, **kwargs):
    return ~x


@dispatch(USub, np.ndarray)
def compute_one(t, x, **kwargs):
    return -x


@dispatch(Selection, np.ndarray)
def compute_one(t, x, **kwargs):
    predicate = compute(t.predicate, {t.child: x})
    return x[predicate]


@dispatch(count, np.ndarray)
def compute_one(t, x, **kwargs):
    return len(x)


@dispatch(nunique, np.ndarray)
def compute_one(t, x, **kwargs):
    return len(np.unique(x))


@dispatch(Reduction, np.ndarray)
def compute_one(t, x, **kwargs):
    return getattr(x, t.symbol)()


@dispatch(Distinct, np.ndarray)
def compute_one(t, x, **kwargs):
    return np.unique(x)


@dispatch(Sort, np.ndarray)
def compute_one(t, x, **kwargs):
    if (t.key in x.dtype.names or
        isinstance(t.key, list) and all(k in x.dtype.names for k in t.key)):
        result = np.sort(x, order=t.key)
    elif key:
        raise NotImplementedError("Sort key %s not supported" % str(t.key))
    else:
        result = np.sort(x)

    if t.ascending == False:
        result = result[::-1]

    return result


@dispatch(Head, np.ndarray)
def compute_one(t, x, **kwargs):
    return x[:t.n]


@dispatch(Label, np.ndarray)
def compute_one(t, x, **kwargs):
    return np.array(x, dtype=[(t.label, x.dtype.type)])


@dispatch(ReLabel, np.ndarray)
def compute_one(t, x, **kwargs):
    types = [x.dtype[i] for i in range(len(x.dtype))]
    return np.array(x, dtype=list(zip(t.columns, types)))


@dispatch(Selection, np.ndarray)
def compute_one(sel, x, **kwargs):
    return x[compute(sel.predicate, {sel.child: x})]


@dispatch(Union, np.ndarray, tuple)
def compute_one(expr, example, children, **kwargs):
    return np.concatenate(list(children), axis=0)


@dispatch(TableExpr, np.ndarray)
def compute_one(t, x, **kwargs):
    from blaze.api.into import into
    from pandas import DataFrame
    df = into(DataFrame(columns=t.child.columns), x)
    return compute_one(t, df, **kwargs)


@dispatch(np.ndarray)
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize
