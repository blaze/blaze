from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from toolz import map, partition_all
import numpy as np
import math
from collections import Iterator

from ..compatibility import builtins
from ..dispatch import dispatch


class ChunkIter(object):
    def __init__(self, seq):
        self.seq = seq

    def __iter__(self):
        return self.seq


class Chunks(ChunkIter):
    def __init__(self, seq, chunksize=1024):
        self.seq = seq

    def __iter__(self):
        return chunks(self.seq)


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}


@dispatch(tuple(reductions), ChunkIter)
def compute_one(expr, c, **kwargs):
    t = TableSymbol('_', schema=expr.child)
    a, b = reductions[type(expr)]

    return compute_one(b(t), [compute_one(a(t), chunk) for chunk in c])

@dispatch(mean, ChunkIter)
def compute_one(expr, c, **kwargs):
    total_sum = 0
    total_count = 0
    for chunk in c:
        if isinstance(chunk, Iterator):
            chunk = list(chunk)
        total_sum += compute_one(expr.child.sum(), chunk)
        total_count += compute_one(expr.child.count(), chunk)

    return total_sum / total_count


@dispatch((Selection, RowWise), ChunkIter)
def compute_one(expr, c, **kwargs):
    return ChunkIter(compute_one(expr, chunk) for chunk in c)


@dispatch(Join, object, ChunkIter)
def compute_one(expr, other, c, **kwargs):
    return ChunkIter(compute_one(expr, other, chunk) for chunk in c)


@dispatch(Join, ChunkIter, object)
def compute_one(expr, c, other, **kwargs):
    return ChunkIter(compute_one(expr, chunk, other) for chunk in c)


@dispatch(Join, Chunks, ChunkIter)
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIter(compute_one(expr, c1, chunk) for chunk in c2)


@dispatch(Join, ChunkIter, Chunks)
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIter(compute_one(expr, chunk, c2) for chunk in c1)


@dispatch(Join, ChunkIter, ChunkIter)
def compute_one(expr, c1, c2, **kwargs):
    raise NotImplementedError("Can not perform chunked join of "
            "two chunked iterators")


@dispatch((list, tuple, Iterator))
def chunks(seq, chunksize=1024):
    return partition_all(chunksize, seq)

