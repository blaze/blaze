"""
Computing in chunks, a meta-backend

Many computations can be done in chunks.  This is useful if the entire dataset
doesn't fit comfortably into memory.

For example, the sum of a very large collection can be computed by taking large
chunks into memory, performing an in-memory sum on each chunk in turn, and then
summing the resulting sums.  E.g.

    @dispatch(sum, ChunkIterator)
    def compute_up(expr, chunks):
        sums = []
        for chunk in chunks:
            sums.append(compute_up(expr, chunk))
        return builtin.sum(sums)

Using tricks like this we can apply the operations from rich in memory backends
like Pandas onto more restricted out-of-core backends like PyTables.
"""

from __future__ import absolute_import, division, print_function

import itertools
from ..expr import (Symbol, Head, Join, Selection, By, Label,
        ElemWise, ReLabel, Distinct, by, min, max, any, all, sum, count, mean,
        nunique, Arithmetic, Broadcast, symbol)
from .core import compute
from toolz import partition_all, curry, concat, first
from collections import Iterator, Iterable
from cytoolz import unique
from datashape import var, isdimension
from datashape.predicates import isscalar
import pandas as pd
import numpy as np

from ..dispatch import dispatch
from ..data.core import DataDescriptor
from ..expr.split import split
from ..expr import Expr

__all__ = ['ChunkIterable', 'ChunkIterator', 'ChunkIndexable', 'get_chunk',
           'chunks', 'into']


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


@dispatch(Expr, ChunkIterator)
def pre_compute(expr, data, scope=None):
    return data


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}


@dispatch(tuple(reductions), ChunkIterator)
def compute_up(expr, c, **kwargs):
    t = symbol('_', expr._child.dshape)
    a, b = reductions[type(expr)]

    return compute_up(b(t), [compute_up(a(t), pre_compute(expr, chunk))
                                for chunk in c])


@dispatch(mean, ChunkIterator)
def compute_up(expr, c, **kwargs):
    total_sum = 0
    total_count = 0
    for chunk in c:
        if isinstance(chunk, Iterator):
            chunk = list(chunk)
        total_sum += compute_up(expr._child.sum(), pre_compute(expr, chunk))
        total_count += compute_up(expr._child.count(), pre_compute(expr, chunk))

    return total_sum / total_count


@dispatch(Head, ChunkIterator)
def compute_up(expr, c, **kwargs):
    c = iter(c)
    df = into(DataFrame, compute_up(expr, next(c)),
              columns=expr.fields)
    for chunk in c:
        if len(df) >= expr.n:
            break
        df2 = into(DataFrame,
                   compute_up(expr._child.head(expr.n - len(df)), chunk),
                   columns=expr.fields)
        df = pd.concat([df, df2], axis=0, ignore_index=True)

    return df


@dispatch((Selection, ElemWise, Label, ReLabel, Arithmetic, Broadcast),
          ChunkIterator)
def compute_up(expr, c, **kwargs):
    return ChunkIterator(compute_up(expr, pre_compute(expr, chunk))
            for chunk in c)


@dispatch(Join, ChunkIterable, (ChunkIterable, ChunkIterator))
def compute_up(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_up(expr, c1, pre_compute(expr, chunk))
            for chunk in c2)


@dispatch(Join, ChunkIterator, ChunkIterable)
def compute_up(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_up(expr, pre_compute(expr, chunk), c2)
            for chunk in c1)


@dispatch(Join, ChunkIterator, ChunkIterator)
def compute_up(expr, c1, c2, **kwargs):
    raise NotImplementedError("Can not perform chunked join of "
            "two chunked iterators")

dict_items = type(dict().items())

@dispatch(Join, (object, tuple, list, Iterator, dict_items, DataDescriptor), ChunkIterator)
def compute_up(expr, other, c, **kwargs):
    return ChunkIterator(compute_up(expr, other, pre_compute(expr, chunk))
            for chunk in c)


@dispatch(Join, ChunkIterator, (tuple, list, object, dict_items, Iterator,
    DataDescriptor))
def compute_up(expr, c, other, **kwargs):
    return ChunkIterator(compute_up(expr, pre_compute(expr, chunk), other)
            for chunk in c)


@dispatch(Distinct, ChunkIterator)
def compute_up(expr, c, **kwargs):
    intermediates = concat(into(Iterator,
                                compute_up(expr, pre_compute(expr, chunk)))
        for chunk in c)
    return unique(intermediates)


@dispatch(nunique, ChunkIterator)
def compute_up(expr, c, **kwargs):
    dist = compute_up(expr._child.distinct(), c)
    return compute_up(expr._child.count(), dist)


@dispatch(By, ChunkIterator)
def compute_up(expr, c, **kwargs):
    (chunkleaf, chunkexpr), (aggleaf, aggexpr) = split(expr._child, expr)

    # Put each chunk into a list, then concatenate
    intermediate = list(concat(into([], compute(chunkexpr, {chunkleaf: chunk}))
                          for chunk in c))

    return compute(aggexpr, {aggleaf: intermediate})


@dispatch(object)
def chunks(seq, chunksize=None):
    """ Produce iterable of chunks from data source

    Chunks consumes a data resource and produces an iterator of data resources.

    Elements of the return iterator should have the following properties:

    *  They should fit in memory
    *  They should cover the dataset
    *  They should not overlap
    *  They should honor order if the underlying dataset is ordered
    *  Has appropriate implementations for ``discover``, ``compute_up``,
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
    *  Has appropriate implementations for ``discover``, ``compute_up``,
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


@dispatch(ChunkIterable)
def discover(c):
    ds = discover(first(c))
    assert isdimension(ds[0])
    return var * ds.subshape[0]


class ChunkList(ChunkIndexable):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)


def compute_chunk(source, chunk, chunk_expr, index):
    part = source[index]
    return compute(chunk_expr, {chunk: part})


@dispatch(Expr, ChunkList)
def compute_down(expr, data, map=map, **kwargs):
    leaf = expr._leaves()[0]

    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr)

    indices = list(range(len(data.data)))

    parts = list(map(curry(compute_chunk, data.data, chunk, chunk_expr),
                     indices))

    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], (Iterable, Iterator)):
        intermediate = concat(parts)

    return compute(agg_expr, {agg: intermediate})


from ..resource import resource
from glob import glob

@resource.register('.*\*.*', priority=14)
def resource_glob(uri, **kwargs):
    uris = sorted(glob(uri))

    first = resource(uris[0], **kwargs)
    if hasattr(first, 'dshape'):
        kwargs['dshape'] = first.dshape
    if hasattr(first, 'schema'):
        kwargs['schema'] = first.schema

    return ChunkList([resource(uri, **kwargs) for uri in uris])
