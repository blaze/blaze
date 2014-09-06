from __future__ import absolute_import, division, print_function

import numpy as np
from blaze.expr import Reduction, Column, Projection, ColumnWise, Selection
from blaze.expr import Distinct, Sort, Head, Label, ReLabel, Union, TableExpr, Sample
from blaze.expr import std, var, count, nunique
from blaze.expr.scalar import BinOp, UnaryOp, USub, Not

from .core import base, compute
from ..dispatch import dispatch
from blaze.api.into import into
from pandas import DataFrame, Series

__all__ = ['np']


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
    if not x.dtype.names and x.shape[1] == len(t.child.columns):
        return x[:, [t.child.columns.index(col) for col in t.columns]]
    raise NotImplementedError()


@dispatch(ColumnWise, np.ndarray)
def compute_one(t, x, **kwargs):
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


@dispatch((std, var), np.ndarray)
def compute_one(t, x, **kwargs):
    return getattr(x, t.symbol)(ddof=t.unbiased)


@dispatch(Distinct, np.ndarray)
def compute_one(t, x, **kwargs):
    return np.unique(x)


@dispatch(Sort, np.ndarray)
def compute_one(t, x, **kwargs):
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
    if x.ndim > 1 or isinstance(x, np.recarray) or x.dtype.fields is not None:
        df = DataFrame(columns=t.child.columns)
    else:
        df = Series(name=t.child.columns[0])
    return compute_one(t, into(df, x), **kwargs)

@dispatch(Sample, np.ndarray)
def compute_one(expr, data, **kwargs):
    """ A small,randomly selected sample of data from the given numpy.ndarray

    Parameters
    ----------
    expr : TableExpr
        The TableExpr that we are calculating over
    data : numpy.ndarray
        The numpy ndarray we are sampling from

    Returns
    -------
    numpy.ndarray
        A new numpy.ndarray

    Notes
    -----
    Each time compute(expression.sample(), ndarray) is called a new, different
    numpy.ndarray should be returned.

    Examples
    --------
    >>> x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
            dtype=[('id', 'i8'), ('name', 'S7'), ('amount', 'i8')])
    >>> t = TableSymbol('t', '{id: int, name: string, amount: int}')
    >>> result=compute(t.sample(2), x)
    >>> assert(len(result) == 2)
    """

    array_len = len(data)
    count=expr.n
    if count > array_len and expr.replacement is False:
        #If we make it here, the user has requested more values than can be returned
        #  So, we need to pare things down.
        #In essence, this now works like a permutation()
        count=array_len

    indices = np.random.choice(array_len, count, replace=expr.replacement)
    result = data.take(indices)

    return result


@dispatch(np.ndarray)
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize
