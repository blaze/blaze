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

from blaze.expr import *
from toolz import partition_all
from collections import Iterator, Iterable
from toolz import concat, first
from cytoolz import unique
from datashape import var, isdimension
import pandas as pd
from ..api.resource import resource
from glob import glob
from pandas import DataFrame
import pandas


from ..dispatch import dispatch
from ..data.core import DataDescriptor
from ..expr.split import split

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
def compute_up(expr, c, **kwargs):
    t = TableSymbol('_', dshape=expr.child.dshape)
    a, b = reductions[type(expr)]

    return compute_up(b(t), [compute_up(a(t), chunk) for chunk in c])


@dispatch(mean, ChunkIterator)
def compute_up(expr, c, **kwargs):
    total_sum = 0
    total_count = 0
    for chunk in c:
        if isinstance(chunk, Iterator):
            chunk = list(chunk)
        total_sum += compute_up(expr.child.sum(), chunk)
        total_count += compute_up(expr.child.count(), chunk)

    return total_sum / total_count


@dispatch(Head, ChunkIterator)
def compute_up(expr, c, **kwargs):
    c = iter(c)
    df = into(DataFrame, compute_up(expr, next(c)),
              columns=expr.columns)
    for chunk in c:
        if len(df) >= expr.n:
            break
        df2 = into(DataFrame,
                   compute_up(expr.child.head(expr.n - len(df)), chunk),
                   columns=expr.columns)
        df = pd.concat([df, df2], axis=0, ignore_index=True)

    return df


@dispatch((Selection, RowWise, Label, ReLabel), ChunkIterator)
def compute_up(expr, c, **kwargs):
    return ChunkIterator(compute_up(expr, chunk) for chunk in c)


@dispatch(Join, ChunkIterable, (ChunkIterable, ChunkIterator))
def compute_up(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_up(expr, c1, chunk) for chunk in c2)


@dispatch(Join, ChunkIterator, ChunkIterable)
def compute_up(expr, c1, c2, **kwargs):
    return ChunkIterator(compute_up(expr, chunk, c2) for chunk in c1)


@dispatch(Join, ChunkIterator, ChunkIterator)
def compute_up(expr, c1, c2, **kwargs):
    raise NotImplementedError("Can not perform chunked join of "
            "two chunked iterators")

dict_items = type(dict().items())

@dispatch(Join, (object, tuple, list, Iterator, dict_items, DataDescriptor), ChunkIterator)
def compute_up(expr, other, c, **kwargs):
    return ChunkIterator(compute_up(expr, other, chunk) for chunk in c)


@dispatch(Join, ChunkIterator, (tuple, list, object, dict_items, Iterator,
    DataDescriptor))
def compute_up(expr, c, other, **kwargs):
    return ChunkIterator(compute_up(expr, chunk, other) for chunk in c)


@dispatch(Distinct, ChunkIterator)
def compute_up(expr, c, **kwargs):
    intermediates = concat(into(Iterator, compute_up(expr, chunk)) for chunk in c)
    return unique(intermediates)


@dispatch(nunique, ChunkIterator)
def compute_up(expr, c, **kwargs):
    dist = compute_up(expr.child.distinct(), c)
    return compute_up(expr.child.count(), dist)


@dispatch(By, ChunkIterator)
def compute_up(expr, c, **kwargs):
    (chunkleaf, chunkexpr), (aggleaf, aggexpr) = split(expr.child, expr)

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


@resource.register('.*\*.*', priority=14)
def resource_glob(uri, skip=None, **kwargs):
    uris = sorted(glob(uri))

    first = resource(uris[0], **kwargs)
    if hasattr(first, 'dshape'):
        kwargs['dshape'] = first.dshape
    if hasattr(first, 'schema'):
        kwargs['schema'] = first.schema

    if skip is None:
        return ChunkList([resource(uri, **kwargs) for uri in uris])

    assert all(isinstance(s, Exception) for s in
               ([skip] if not isinstance(skip, Iterable) else skip)), \
        'skip parameter must consist of subclasses of Exception'

    resources = []
    for uri in uris:
        try:
            r = resource(uri, **kwargs)
        except skip:
            pass
        else:
            resources.append(r)
    return ChunkList(resources)
