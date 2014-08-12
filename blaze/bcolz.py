from __future__ import absolute_import, division, print_function

import bcolz
from bcolz import carray, ctable
import numpy as np
from pandas import DataFrame
from collections import Iterator
from toolz import partition_all

from .dispatch import dispatch
from .compute.bcolz import *


__all__ = ['into']


@dispatch(type, (ctable, carray))
def into(a, b, **kwargs):
    f = into.resolve((a, type(b)))
    return f(a, b, **kwargs)

@dispatch((tuple, set, list), (ctable, carray))
def into(o, b):
    return into(o, into(np.ndarray(0), b))


@dispatch(np.ndarray, (ctable, carray))
def into(a, b):
    return b[:]


@dispatch(ctable, np.ndarray)
def into(a, b, **kwargs):
    return ctable(b, **kwargs)


@dispatch(carray, np.ndarray)
def into(a, b, **kwargs):
    return carray(b, **kwargs)


def fix_len_string_filter(ser):
    """ Convert object strings to fixed length, pass through others """
    if ser.dtype == np.dtype('O'):
        return np.asarray(list(ser))
    else:
        return np.asarray(ser)


@dispatch(ctable, DataFrame)
def into(a, df, **kwargs):
    return ctable([fix_len_string_filter(df[c]) for c in df.columns],
                      names=list(df.columns), **kwargs)


@dispatch(carray, (tuple, list))
def into(a, b, **kwargs):
    x = into(np.ndarray(0), b)
    return into(a, x, **kwargs)


@dispatch(ctable, (tuple, list))
def into(a, b, **kwargs):
    if isinstance(b[0], (tuple, list)):
        return ctable([into(np.ndarray(0), c2) for c2 in zip(*b)], **kwargs)
    else:
        return ctable([into(np.ndarray(0), b)], **kwargs)


@dispatch((carray, ctable), Iterator)
def into(a, b, **kwargs):
    chunks = partition_all(1024, b)
    chunk = next(chunks)
    a = ctable([into(np.ndarray(0), c2) for c2 in zip(*chunk)], **kwargs)
    for chunk in chunks:
        a.append(list(zip(*chunk)))
    a.flush()
    return a

@dispatch(DataFrame, ctable)
def into(a, b, columns=None, schema=None):
    if not columns and schema:
        columns = dshape(schema)[0].names
    return DataFrame.from_items(((column, b[column][:]) for column in
                                    sorted(b.names)),
                                orient='columns',
                                columns=columns)


from .compute.chunks import ChunkIter

@dispatch((carray, ctable), ChunkIter)
def into(a, b, **kwargs):
    b = iter(b)
    a = into(a, next(b), **kwargs)
    for chunk in b:
        a.append(into(np.ndarray(0), chunk))
    a.flush()
    return a


from blaze.data.core import DataDescriptor
@dispatch(DataDescriptor, (ctable, carray))
def into(a, b, **kwargs):
    a.extend_chunks(chunks(b))
    return a
