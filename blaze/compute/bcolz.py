from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
import bcolz
from toolz import map
import numpy as np
import math
from .chunks import Chunks, ChunkIter

from ..compatibility import builtins
from ..dispatch import dispatch


@dispatch(bcolz.ctable)
def discover(t):
    return datashape.from_numpy(t.shape, t.dtype)


@dispatch(Selection, (bcolz.ctable, bcolz.carray))
def compute_one(sel, t, **kwargs):
    s = eval_str(sel.predicate.expr)
    try:
        t.where(s)
        return ChunkIter(t.whereblocks(s))
    except (NotImplementedError, NameError): # numexpr may not be able to handle the predicate
        return ChunkIter(compute_one(sel, chunk) for chunk in chunks(t))


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


@dispatch(count, bcolz.ctable)
def compute_one(c, t, **kwargs):
    return len(t)


@dispatch(mean, bcolz.carray)
def compute_one(expr, ba, **kwargs):
    return ba.sum() / ba.len


@dispatch(var, bcolz.carray)
def compute_one(expr, ba, chunksize=2**20, **kwargs):

    E_X_2 = builtins.sum((chunk**2).sum() / chunksize for chunk in chunks(ba))
    E_X_2 *= float(chunksize) * math.ceil(ba.len / float(chunksize)) / ba.len

    E_2_X = float(ba.sum()) / ba.len

    return E_X_2 - E_2_X**2


@dispatch(std, bcolz.carray)
def compute_one(expr, ba, **kwargs):
    return math.sqrt(compute_one(expr.child.var(), ba, **kwargs))


@dispatch((RowWise, Distinct, Reduction, By, count, Label, ReLabel, nunique), (bcolz.carray, bcolz.ctable))
def compute_one(c, t, **kwargs):
    return compute_one(c, Chunks(t), **kwargs)


@dispatch((bcolz.carray, bcolz.ctable))
def chunks(b, chunksize=2**15):
    start = 0
    n = b.len
    while start < n:
        yield b[start:start + chunksize]
        start += chunksize
