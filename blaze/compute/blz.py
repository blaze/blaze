from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
import blz
from toolz import map
import numpy as np
import math
from .chunks import Chunks, ChunkIter

from ..compatibility import builtins
from ..dispatch import dispatch

@dispatch(blz.btable)
def discover(t):
    return datashape.from_numpy(t.shape, t.dtype)


@dispatch(Selection, blz.btable)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return map(tuple, t.where(s))


@dispatch(Selection, blz.barray)
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    return t.where(s)


@dispatch(Head, (blz.barray, blz.btable))
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


@dispatch((RowWise, Distinct, Reduction, By, count, Label, ReLabel, nunique),  (blz.barray, blz.btable))
def compute_one(c, t, **kwargs):
    return compute_one(c, Chunks(t), **kwargs)


@dispatch(mean, blz.barray)
def compute_one(expr, ba, **kwargs):
    return ba.sum() / ba.len


@dispatch(var, blz.barray)
def compute_one(expr, ba, **kwargs):
    E_X_2 = builtins.sum((chunk**2).sum() / 1024 for chunk in chunks(ba))
    E_X_2 *= 1024. * math.ceil(ba.len / 1024.) / ba.len

    E_2_X = float(ba.sum()) / ba.len

    return E_X_2 - E_2_X**2


@dispatch(std, blz.barray)
def compute_one(expr, ba, **kwargs):
    return math.sqrt(compute_one(expr.child.var(), ba, **kwargs))


@dispatch((blz.barray, blz.btable))
def chunks(b, chunksize=1024):
    start = 0
    n = b.len
    while start < n:
        yield b[start:start + chunksize]
        start += chunksize
