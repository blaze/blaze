from __future__ import absolute_import, division, print_function

from blaze.expr import Selection, Head, Column, Projection, ReLabel, RowWise
from blaze.expr import Label, Distinct, By, Reduction, Like
from blaze.expr import std, var, count, mean, nunique, sum
from blaze.expr import eval_str

import datashape
import bcolz
import math
from .chunks import ChunkIndexable


from ..compatibility import builtins
from ..dispatch import dispatch

__all__ = ['bcolz']


@dispatch(bcolz.ctable)
def discover(t):
    return datashape.from_numpy(t.shape, t.dtype)


@dispatch(Selection, (bcolz.ctable, bcolz.carray))
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    try:
        return t.where(s)
    except (NotImplementedError, NameError):
        # numexpr may not be able to handle the predicate
        return compute_one(sel, iter(t), **kwargs)


@dispatch(Head, (bcolz.carray, bcolz.ctable))
def compute_one(h, t, **kwargs):
    return t[:h.n]


@dispatch(Column, bcolz.ctable)
def compute_one(c, t, **kwargs):
    return t[c.column]


@dispatch(Projection, bcolz.ctable)
def compute_one(p, t, **kwargs):
    return t[p.columns]


@dispatch(sum, (bcolz.carray, bcolz.ctable))
def compute_one(expr, t, **kwargs):
    return t.sum()


@dispatch(count, (bcolz.ctable, bcolz.carray))
def compute_one(c, t, **kwargs):
    return len(t)


@dispatch(mean, bcolz.carray)
def compute_one(expr, ba, **kwargs):
    return ba.sum() / ba.len


@dispatch(var, bcolz.carray)
def compute_one(expr, ba, chunksize=2 ** 20, **kwargs):
    n = ba.len
    E_X_2 = builtins.sum((chunk * chunk).sum() for chunk in chunks(ba))
    E_X = float(ba.sum())
    return (E_X_2 - (E_X * E_X) / n) / (n - expr.unbiased)


@dispatch(std, bcolz.carray)
def compute_one(expr, ba, **kwargs):
    result = compute_one(expr.child.var(unbiased=expr.unbiased), ba, **kwargs)
    return math.sqrt(result)


@dispatch((ReLabel, Label), (bcolz.carray, bcolz.ctable))
def compute_one(expr, b, **kwargs):
    raise NotImplementedError()


@dispatch((RowWise, Distinct, By, nunique, Like), bcolz.ctable)
def compute_one(c, t, **kwargs):
    return compute_one(c, iter(t), **kwargs)


@dispatch(nunique, bcolz.carray)
def compute_one(expr, data, **kwargs):
    return len(set(data))


@dispatch(Reduction, (bcolz.carray, bcolz.ctable))
def compute_one(expr, data, **kwargs):
    return compute_one(expr, ChunkIndexable(data), **kwargs)


@dispatch((bcolz.carray, bcolz.ctable))
def chunks(b, chunksize=2**15):
    start = 0
    n = b.len
    while start < n:
        yield b[start:start + chunksize]
        start += chunksize


@dispatch((bcolz.carray, bcolz.ctable), int)
def get_chunk(b, i, chunksize=2**15):
    start = chunksize * i
    stop = chunksize * (i + 1)
    return b[start:stop]
