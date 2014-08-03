from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
import blz
from toolz import map
import numpy as np

from ..compatibility import builtins
from ..dispatch import dispatch

@dispatch(blz.btable)
def discover(t):
    return datashape.from_numpy(t.shape, t.dtype)


@dispatch(Selection, blz.btable)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return map(tuple, t.where(s))


@dispatch(Head, blz.btable)
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(Column, blz.btable)
def compute_one(c, t, **kwargs):
    return t[c.column]


@dispatch(Projection, blz.btable)
def compute_one(p, t, **kwargs):
    return t[p.columns]


@dispatch(sum, blz.btable)
def compute_one(expr, t, **kwargs):
    return t.sum()


@dispatch(count, blz.btable)
def compute_one(c, t, **kwargs):
    return len(t)


reductions = {min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}

@dispatch(tuple(reductions), blz.barray)
def compute_one(expr, col, **kwargs):
    t = TableSymbol('_', schema=expr.child)
    a, b = reductions[type(expr)]

    return compute_one(b(t),
                       [compute_one(a(t), chunk) for chunk in chunks(col)])


@dispatch(nunique, (blz.barray, blz.btable))
def compute_one(expr, bt, **kwargs):
    t = TableSymbol('_', schema=expr.child)
    # TODO: replace this with union operation
    union = np.vstack([compute_one(t.distinct(), chunk) for chunk in chunks(bt)])
    return compute_one(t.nunique(), union)


@dispatch(Distinct, (blz.barray, blz.btable))
def compute_one(expr, bt, **kwargs):
    t = TableSymbol('_', schema=expr.child)
    union = np.vstack([compute_one(t.distinct(), chunk) for chunk in chunks(bt)])
    return compute_one(t.distinct(), union)

@dispatch(Expr, blz.btable)
def compute_one(e, t, **kwargs):
    return compute_one(e, iter(t), **kwargs)


@dispatch((blz.barray, blz.btable))
def chunks(b, chunksize=1024):
    start = 0
    n = b.len
    while start < n:
        yield b[start:start + chunksize]
        start += chunksize
