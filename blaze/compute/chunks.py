"""
Computing in chunks, a meta-backend

Many computations can be done in chunks.  This is useful if the entire dataset
doesn't fit comfortably into memory.

For example, the sum of a very large collection can be computed by taking large
chunks into memory, performing an in-memory sum on each chunk in turn, and then
summing the resulting sums.  E.g.

    @dispatch(sum, ChunkIter)
    def compute_one(expr, chunks):
        sums = []
        for chunk in chunks:
            sums.append(compute_one(expr, chunk))
        return builtin.sum(sums)

Using tricks like this we can apply the operations from rich in memory backends
like Pandas onto more restricted out-of-core backends like PyTables.
"""

from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from toolz import map, partition_all, reduce
import numpy as np
import math
from collections import Iterator
from toolz import concat
from cytoolz import unique

from ..compatibility import builtins
from ..dispatch import dispatch
from .core import compute


class ChunkIter(object):
    def __init__(self, seq):
        self.seq = seq

    def __iter__(self):
        return self.seq


class Chunks(ChunkIter):
    def __init__(self, seq, **kwargs):
        self.seq = seq
        self.kwargs = kwargs

    def __iter__(self):
        return chunks(self.seq, **self.kwargs)


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


@dispatch(Head, ChunkIter)
def compute_one(expr, c, **kwargs):
    c = iter(c)
    n = 0
    cs = []
    for chunk in c:
        cs.append(chunk)
        n += len(chunk)
        if n >= expr.n:
            break

    if not cs:
        return []

    if len(cs) == 1:
        return compute_one(expr, cs[0])

    t1 = TableSymbol('t1', expr.schema)
    t2 = TableSymbol('t2', expr.schema)
    binop = lambda a, b: compute(union(t1, t2), {t1: a, t2: b})
    u = reduce(binop, cs)

    return compute_one(expr, u)


@dispatch((Selection, RowWise), ChunkIter)
def compute_one(expr, c, **kwargs):
    return ChunkIter(compute_one(expr, chunk) for chunk in c)


@dispatch(Join, Chunks, (Chunks, ChunkIter))
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIter(compute_one(expr, c1, chunk) for chunk in c2)


@dispatch(Join, ChunkIter, Chunks)
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIter(compute_one(expr, chunk, c2) for chunk in c1)


@dispatch(Join, ChunkIter, ChunkIter)
def compute_one(expr, c1, c2, **kwargs):
    raise NotImplementedError("Can not perform chunked join of "
            "two chunked iterators")


@dispatch(Join, object, ChunkIter)
def compute_one(expr, other, c, **kwargs):
    return ChunkIter(compute_one(expr, other, chunk) for chunk in c)


@dispatch(Join, ChunkIter, object)
def compute_one(expr, c, other, **kwargs):
    return ChunkIter(compute_one(expr, chunk, other) for chunk in c)


@dispatch(Distinct, ChunkIter)
def compute_one(expr, c, **kwargs):
    intermediates = concat(into([], compute_one(expr, chunk)) for chunk in c)
    return unique(intermediates)


@dispatch(nunique, ChunkIter)
def compute_one(expr, c, **kwargs):
    dist = compute_one(expr.child.distinct(), c)
    return compute_one(expr.child.count(), dist)


@dispatch(By, ChunkIter)
def compute_one(expr, c, **kwargs):
    if not isinstance(expr.apply, tuple(reductions)):
        raise NotImplementedError("Chunked split-apply-combine only "
                "implemented for simple reductions")

    a, b = reductions[type(expr.apply)]

    perchunk = by(expr.child, expr.grouper, a(expr.apply.child))

    # Put each chunk into a list, then concatenate
    intermediate = concat(into([], compute_one(perchunk, chunk))
                          for chunk in c)

    # Form computation to do on the concatenated union
    t = TableSymbol('_chunk', perchunk.schema)

    apply_cols = expr.apply.dshape[0].names
    if expr.apply.child.iscolumn:
        apply_cols = apply_cols[0]

    group = by(t,
               t[expr.grouper.columns],
               b(t[apply_cols]))

    return compute_one(group, intermediate)


@dispatch((list, tuple, Iterator))
def chunks(seq, chunksize=1024):
    return partition_all(chunksize, seq)


@dispatch((list, tuple), ChunkIter)
def into(a, b):
    return type(a)(concat((into(a, chunk) for chunk in b)))


from pandas import DataFrame
import pandas

@dispatch(DataFrame, ChunkIter)
def into(df, b, **kwargs):
    chunks = [into(df, chunk, **kwargs) for chunk in b]
    return pandas.concat(chunks, ignore_index=True)
