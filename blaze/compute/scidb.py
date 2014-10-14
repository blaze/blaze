from __future__ import absolute_import, division, print_function

import numpy as np

from ..expr import Reduction, Column, Projection, ColumnWise, Selection
from blaze.expr import Distinct, Sort, Head, Label, ReLabel, Union, TableExpr, By
from blaze.expr import std, var, count, nunique
from blaze.expr.scalar import BinOp, UnaryOp, USub, Not
from blaze.dispatch import dispatch as dispatch
from blaze.api.into import into
from blaze.compute.core import base, compute

from pandas import DataFrame

from scidbpy import SciDBArray
from scidbpy.aggregation import GroupBy
from scidbpy.utils import new_attribute_label


@dispatch(Column, SciDBArray)
def compute_up(c, x, **kwargs):
    return x[c.column]


@dispatch(Projection, SciDBArray)
def compute_up(t, x, **kwargs):
    return x[t.columns]


@dispatch(ColumnWise, SciDBArray)
def compute_up(t, x, **kwargs):
    d = dict((t.child[c].scalar_symbol, x[c]) for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(BinOp, SciDBArray, (SciDBArray, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, base, SciDBArray)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, SciDBArray)
def compute_up(t, x, **kwargs):
    return getattr(x.interface, t.symbol)(x)


@dispatch((USub, Not), SciDBArray)
def compute_up(t, x, **kwargs):
    return t.op(x)


@dispatch(Selection, SciDBArray)
def compute_up(t, x, **kwargs):
    predicate = compute(t.predicate, {t.child: x})
    return x[predicate]


@dispatch(count, SciDBArray)
def compute_up(t, x, **kwargs):
    return x.count()


@dispatch(nunique, SciDBArray)
def compute_up(t, x, **kwargs):
    return x.approxdc()


@dispatch(Reduction, SciDBArray)
def compute_up(t, x, **kwargs):
    return getattr(x, t.symbol)()


@dispatch(var, SciDBArray)
def compute_up(t, x, **kwargs):
    result = x.var()
    N = x.size

    # scidb returns unbiased variance
    if not t.unbiased:
        result *= (N - 1) / N

    return result


@dispatch(std, SciDBArray)
def compute_up(t, x, **kwargs):
    result = x.std()
    N = x.size

    # scidb returns unbiased std
    if not t.unbiased:
        result *= np.sqrt((N - 1) / N)

    return result


@dispatch(Distinct, SciDBArray)
def compute_up(t, x, **kwargs):
    return x.uniq()


@dispatch(Sort, SciDBArray)
def compute_up(t, x, **kwargs):
    f = x.afl
    key = t.key or []

    if not isinstance(key, list):
        key = [key]

    if all(k in x.att_names for k in key):
        args = key
        if not t.ascending:
            args = ['%s desc' % k for k in key]
        return f.sort(x, *args)

    raise NotImplementedError("Sort key %s not supported" % str(t.key))


@dispatch(Head, SciDBArray)
def compute_up(t, x, **kwargs):
    return x[:t.n]


@dispatch(Label, SciDBArray)
def compute_up(t, x, **kwargs):
    return x.attribute_rename(x.att_names[0], t.label)


@dispatch(ReLabel, SciDBArray)
def compute_up(t, x, **kwargs):
    types = [x.dtype[i] for i in range(len(x.dtype))]
    old = x.att_names
    return x.attribute_rename(*(item
                                for oldnew in zip(old, t.columns)
                                for item in oldnew))


@dispatch(Selection, SciDBArray)
def compute_up(sel, x, **kwargs):
    return x[compute(sel.predicate, {sel.child: x})]


@dispatch(Union, SciDBArray, tuple)
def compute_up(expr, example, children, **kwargs):
    return example.interface.concatenate(list(children), axis=1)


@dispatch(SciDBArray)
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize


@dispatch(DataFrame, SciDBArray)
def into(df, arr):
    return arr.todataframe()


@dispatch(list, SciDBArray)
def into(df, arr):
    return arr.tolist()


def _to_scidb_syntax(expr):
    return expr


@dispatch(By, SciDBArray)
def compute_up(t, x, **kwargs):

    # build the groups
    grpr = t.grouper
    if isinstance(grpr, ColumnWise):
        # apply columnwise operations to table
        att = new_attribute_label('cond', x)
        x[att] = _to_scidb_syntax(grpr.expr)
        grpr = att
    elif isinstance(grpr, Column):
        grpr = grpr.column
    else:
        raise NotImplementedError()

    groups = x.groupby(grpr)

    return compute(t.apply, groups)


@dispatch(Column, GroupBy)
def compute_up(c, g, **kwargs):
    return g[c.column]


@dispatch(Reduction, GroupBy)
def compute_up(r, g, **kwargs):
    mappings = dict((c, "{0}({1})".format(r.symbol, c))
                    for c in g.columns)
    return g.aggregate(mappings)
