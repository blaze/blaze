"""
Computing in chunks, a meta-backend

Many computations can be done in chunks.  This is useful if the entire dataset
doesn't fit comfortably into memory.

For example, the sum of a very large collection can be computed by taking large
chunks into memory, performing an in-memory sum on each chunk in turn, and then
summing the resulting sums.  E.g.

    @dispatch(sum, ChunkIterator)
    def compute_one(expr, chunks):
        sums = []
        for chunk in chunks:
            sums.append(compute_one(expr, chunk))
        return builtin.sum(sums)

Using tricks like this we can apply the operations from rich in memory backends
like Pandas onto more restricted out-of-core backends like PyTables.
"""

from __future__ import absolute_import, division, print_function

from blaze.expr import *
from toolz import map, partition_all, reduce
import numpy as np
import math
from collections import Iterator
from toolz import concat
from cytoolz import unique

from ..compatibility import builtins
from ..dispatch import dispatch
from .core import compute
from ..data.core import DataDescriptor

__all__ = ['ChunkIterable', 'ChunkIterator', 'ChunkIndexable', 'get_chunk', 'chunks', 'into']

class ChunkIterator(object):
    def __init__(self, seq):
        self.seq = iter(seq)

    def __iter__(self):
        return self.seq

    def __next__(self):
        return next(self.seq)


class ChunkIterable(ChunkIterator):
    def __init__(self, seq, **kwargs):
        self.seq = seq
        self.kwargs = kwargs

    def __iter__(self):
        return chunks(self.seq, **self.kwargs)


class ChunkIndexable(ChunkIterable):
    def __init__(self, seq, **kwargs):
        self.seq = seq
        self.kwargs = kwargs

    def __getitem__(self, key):
        return get_chunk(self.seq, key, **self.kwargs)

    def __iter__(self):
        try:
            cs = chunks(self.seq, **self.kwargs)
        except NotImplementedError:
            cs = None
        if cs:
            for c in cs:
                yield c
        else:
            for i in itertools.count(0):
                try:
                    yield get_chunk(self.seq, i, **self.kwargs)
                except IndexError:
                    raise StopIteration()

reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}


@dispatch(tuple(reductions), ChunkIterator)
def compute_one(expr, c, **kwargs):
    t = TableSymbol('_', dshape=expr.child)
    a, b = reductions[type(expr)]

    return compute_one(b(t), [compute_one(a(t), chunk) for chunk in c])


@dispatch(mean, ChunkIterator)
def compute_one(expr, c, **kwargs):
    total_sum = 0
    total_count = 0
    for chunk in c:
        if isinstance(chunk, Iterator):
            chunk = list(chunk)
        total_sum += compute_one(expr.child.sum(), chunk)
        total_count += compute_one(expr.child.count(), chunk)

    return total_sum / total_count


@dispatch(Head, ChunkIterator)
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


@dispatch((Selection, RowWise, Label, ReLabel), ChunkIterator)
def compute_one(expr, c, **kwargs):
    return ChunkIterator(compute_one(expr, chunk) for chunk in c)


@dispatch(Join, ChunkIterable, (ChunkIterable, ChunkIterator))
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_one(expr, c1, chunk) for chunk in c2)


@dispatch(Join, ChunkIterator, ChunkIterable)
def compute_one(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_one(expr, chunk, c2) for chunk in c1)


@dispatch(Join, ChunkIterator, ChunkIterator)
def compute_one(expr, c1, c2, **kwargs):
    raise NotImplementedError("Can not perform chunked join of "
            "two chunked iterators")

dict_items = type(dict().items())

@dispatch(Join, (object, tuple, list, Iterator, dict_items, DataDescriptor), ChunkIterator)
def compute_one(expr, other, c, **kwargs):
    return ChunkIterator(compute_one(expr, other, chunk) for chunk in c)


@dispatch(Join, ChunkIterator, (tuple, list, object, dict_items, Iterator,
    DataDescriptor))
def compute_one(expr, c, other, **kwargs):
    return ChunkIterator(compute_one(expr, chunk, other) for chunk in c)


@dispatch(Distinct, ChunkIterator)
def compute_one(expr, c, **kwargs):
    intermediates = concat(into([], compute_one(expr, chunk)) for chunk in c)
    return unique(intermediates)


@dispatch(nunique, ChunkIterator)
def compute_one(expr, c, **kwargs):
    dist = compute_one(expr.child.distinct(), c)
    return compute_one(expr.child.count(), dist)


@dispatch(By, ChunkIterator)
def compute_one(expr, c, **kwargs):
    if not isinstance(expr.apply, tuple(reductions)):
        raise NotImplementedError("Chunked split-apply-combine only "
                "implemented for simple reductions")

    a, b = reductions[type(expr.apply)]

    perchunk = by(expr.grouper, a(expr.apply.child))

    # Put each chunk into a list, then concatenate
    intermediate = concat(into([], compute_one(perchunk, chunk))
                          for chunk in c)

    # Form computation to do on the concatenated union
    t = TableSymbol('_chunk', perchunk.schema)

    apply_cols = expr.apply.dshape[0].names
    if expr.apply.child.iscolumn:
        apply_cols = apply_cols[0]

    group = by(t[expr.grouper.columns],
               b(t[apply_cols]))

    return compute_one(group, intermediate)


@dispatch(object)
def chunks(seq, chunksize=None):
    """ Produce iterable of chunks from data source

    Chunks consumes a data resource and produces an iterator of data resources.

    Elements of the return iterator should have the following properties:

    *  They should fit in memory
    *  They should cover the dataset
    *  They should not overlap
    *  They should honor order if the underlying dataset is ordered
    *  Has appropriate implementations for ``discover``, ``compute_one``,
       and  ``into``

    Example
    -------

    >>> seq = chunks(range(1000), chunksize=3)
    >>> next(seq)
    (0, 1, 2)

    >>> next(seq)
    (3, 4, 5)

    See Also
    --------

    blaze.compute.chunks.chunk
    """
    raise NotImplementedError("chunks not implemented on type %s" %
            type(seq).__name__)


@dispatch(object, int)
def get_chunk(data, i, chunksize=None):
    """ Get the ``i``th chunk from a data resource

    Returns a single chunk of the data

    Return chunks should have the following properties

    *  They should fit in memory
    *  They should cover the dataset
    *  They should not overlap
    *  They should honor order if the underlying dataset is ordered
    *  Has appropriate implementations for ``discover``, ``compute_one``,
       and  ``into``

    Example
    -------

    >>> data = list(range(1000))
    >>> get_chunk(data, 0, chunksize=3)
    [0, 1, 2]

    >>> get_chunk(data, 3, chunksize=3)
    [9, 10, 11]


    See Also
    --------

    blaze.compute.chunks.chunk
    """
    raise NotImplementedError("chunk not implemented on type %s" %
            type(data).__name__)


@dispatch((list, tuple, Iterator))
def chunks(seq, chunksize=1024):
    return partition_all(chunksize, seq)

@dispatch((list, tuple), int)
def get_chunk(seq, i, chunksize=1024):
    start = chunksize * i
    stop = chunksize * (i + 1)
    return seq[start:stop]

@dispatch((list, tuple), ChunkIterator)
def into(a, b):
    return type(a)(concat((into(a, chunk) for chunk in b)))


from pandas import DataFrame
import pandas

@dispatch(DataFrame, ChunkIterator)
def into(df, b, **kwargs):
    chunks = [into(df, chunk, **kwargs) for chunk in b]
    return pandas.concat(chunks, ignore_index=True)
