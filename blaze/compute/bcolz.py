from __future__ import absolute_import, division, print_function

from blaze.expr import Selection, Head, Field, Projection, ReLabel, ElemWise
from blaze.expr import Label, Distinct, By, Reduction, Like, Slice
from blaze.expr import std, var, count, mean, nunique, sum
from blaze.expr import eval_str, Expr
from blaze.expr.optimize import lean_projection

import datashape
import bcolz
import math
from .chunks import ChunkIndexable


from ..compatibility import builtins
from ..dispatch import dispatch

__all__ = ['bcolz']

COMFORTABLE_MEMORY_SIZE = 1e9


@dispatch(bcolz.ctable)
def discover(data):
    return datashape.from_numpy(data.shape, data.dtype)


@dispatch(Selection, (bcolz.ctable, bcolz.carray))
def compute_up(expr, data, **kwargs):
    if data.nbytes < COMFORTABLE_MEMORY_SIZE:
        return compute_up(expr, data[:], **kwargs)
    s = eval_str(expr.predicate.expr)
    try:
        return data.where(s)
    except (NotImplementedError, NameError):
        # numexpr may not be able to handle the predicate
        return compute_up(expr, iter(data), **kwargs)


@dispatch(Head, (bcolz.carray, bcolz.ctable))
def compute_up(expr, data, **kwargs):
    return data[:expr.n]


@dispatch(Field, bcolz.ctable)
def compute_up(expr, data, **kwargs):
    return data[expr._name]


@dispatch(Projection, bcolz.ctable)
def compute_up(expr, data, **kwargs):
    return data[expr.fields]


@dispatch(sum, (bcolz.carray, bcolz.ctable))
def compute_up(expr, data, **kwargs):
    return data.sum()


@dispatch(count, (bcolz.ctable, bcolz.carray))
def compute_up(expr, data, **kwargs):
    return len(data)


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
def compute_up(expr, data, **kwargs):
    if data.nbytes < COMFORTABLE_MEMORY_SIZE:
        return compute_up(expr, data[:], **kwargs)
    return compute_up(expr, iter(data), **kwargs)


@dispatch(nunique, bcolz.carray)
def compute_up(expr, data, **kwargs):
    return len(set(data))


@dispatch(Reduction, (bcolz.carray, bcolz.ctable))
def compute_up(expr, data, **kwargs):
    if data.nbytes < COMFORTABLE_MEMORY_SIZE:
        return compute_up(expr, data[:], **kwargs)
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


@dispatch(Expr, bcolz.ctable)
def optimize(expr, _):
    return lean_projection(expr)
