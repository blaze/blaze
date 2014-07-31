from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from ..dispatch import dispatch
import bcolz
from toolz import map


@dispatch(bcolz.ctable)
def discover(t):
    return datashape.from_numpy(t.shape, t.dtype)


@dispatch(Selection, bcolz.ctable)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return map(tuple, t.where(s))


@dispatch(Head, bcolz.ctable)
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(Column, bcolz.ctable)
def compute_one(c, t, **kwargs):
    return t[c.column]


@dispatch(Projection, bcolz.ctable)
def compute_one(p, t, **kwargs):
    return t[p.columns]


@dispatch(sum, bcolz.ctable)
def compute_one(expr, t, **kwargs):
    return t.sum()


@dispatch(count, bcolz.ctable)
def compute_one(c, t):
    return len(t)


@dispatch(Expr, bcolz.ctable)
def compute_one(e, t, **kwargs):
    return compute_one(e, iter(t), **kwargs)
