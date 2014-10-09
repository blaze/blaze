from __future__ import absolute_import, division, print_function

from blaze.expr import Selection, Head, Field, Projection, ReLabel, ElemWise
from blaze.expr import Label, Distinct, By, Reduction, Like, Slice
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
def compute_up(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    try:
        return t.where(s)
    except (NotImplementedError, NameError):
        # numexpr may not be able to handle the predicate
        return compute_up(sel, iter(t), **kwargs)


@dispatch(Head, (bcolz.carray, bcolz.ctable))
def compute_up(h, t, **kwargs):
    return t[:h.n]


@dispatch(Field, bcolz.ctable)
def compute_up(c, t, **kwargs):
    return t[c._name]


@dispatch(Projection, bcolz.ctable)
def compute_up(p, t, **kwargs):
    return t[p.fields]


@dispatch(sum, (bcolz.carray, bcolz.ctable))
def compute_up(expr, t, **kwargs):
    return t.sum()


@dispatch(count, (bcolz.ctable, bcolz.carray))
def compute_up(c, t, **kwargs):
    return len(t)


@dispatch(mean, bcolz.carray)
def compute_up(expr, ba, **kwargs):
    return ba.sum() / ba.len


@dispatch(var, bcolz.carray)
def compute_up(expr, ba, chunksize=2**20, **kwargs):
    n = ba.len
    E_X_2 = builtins.sum((chunk * chunk).sum() for chunk in chunks(ba))
    E_X = float(ba.sum())
    return (E_X_2 - (E_X * E_X) / n) / (n - expr.unbiased)


@dispatch(std, bcolz.carray)
def compute_up(expr, ba, **kwargs):
    result = compute_up(expr._child.var(unbiased=expr.unbiased), ba, **kwargs)
    return math.sqrt(result)


@dispatch((ReLabel, Label), (bcolz.carray, bcolz.ctable))
def compute_up(expr, b, **kwargs):
    raise NotImplementedError()


@dispatch((ElemWise, Distinct, By, nunique, Like), bcolz.ctable)
def compute_up(c, t, **kwargs):
    return compute_up(c, iter(t), **kwargs)


@dispatch(nunique, bcolz.carray)
def compute_up(expr, data, **kwargs):
    return len(set(data))


@dispatch(Reduction, (bcolz.carray, bcolz.ctable))
def compute_up(expr, data, **kwargs):
    return compute_up(expr, ChunkIndexable(data), **kwargs)


@dispatch(Slice, (bcolz.carray, bcolz.ctable))
def compute_up(expr, x, **kwargs):
    return x[expr.index]


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
