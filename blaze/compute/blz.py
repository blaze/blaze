from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from ..dispatch import dispatch
import blz
from toolz import map


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
def compute_one(c, t):
    return len(t)


@dispatch(Expr, blz.btable)
def compute_one(e, t, **kwargs):
    return compute_one(e, iter(t), **kwargs)
