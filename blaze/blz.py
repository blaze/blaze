from __future__ import absolute_import, division, print_function

import blz
import numpy as np
from pandas import DataFrame
from collections import Iterator
from toolz import partition_all

from .dispatch import dispatch
from .compute.blz import *


__all__ = ['into', 'chunks']


@dispatch((tuple, set, list, type, object), (blz.btable, blz.barray))
def into(o, b):
    return into(o, into(np.ndarray(0), b))


@dispatch(np.ndarray, (blz.btable, blz.barray))
def into(a, b):
    return b[:]


@dispatch((blz.barray, blz.btable), np.ndarray)
def into(a, b, **kwargs):
    return blz.btable(b, **kwargs)


def fix_len_string_filter(ser):
    """ Convert object strings to fixed length, pass through others """
    if ser.dtype == np.dtype('O'):
        return np.asarray(list(ser))
    else:
        return np.asarray(ser)


@dispatch(blz.btable, DataFrame)
def into(a, df, **kwargs):
    return blz.btable([fix_len_string_filter(df[c]) for c in df.columns],
                      names=list(df.columns), **kwargs)


@dispatch((blz.barray, blz.btable), (tuple, list))
def into(a, b, **kwargs):
    return blz.btable(list(zip(*b)), **kwargs)


@dispatch((blz.barray, blz.btable), Iterator)
def into(a, b, **kwargs):
    chunks = partition_all(1024, b)
    chunk = next(chunks)
    a = blz.btable(list(zip(*chunk)), **kwargs)
    for chunk in chunks:
        a.append(list(zip(*chunk)))
    return a


@dispatch(DataFrame, blz.btable)
def into(a, b, columns=None, schema=None):
    if not columns and schema:
        columns = dshape(schema)[0].names
    return DataFrame.from_items(((column, b[column][:]) for column in
                                    sorted(b.names)),
                                orient='columns',
                                columns=columns)


from .compute.chunks import ChunkIter

@dispatch((blz.barray, blz.btable), ChunkIter)
def into(a, b, **kwargs):
    b = iter(b)
    a = into(a, next(b), **kwargs)
    for chunk in b:
        a.append(into(np.ndarray(0), chunk))
    return a


@dispatch((blz.barray, blz.btable))
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize
