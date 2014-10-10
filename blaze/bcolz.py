from __future__ import absolute_import, division, print_function

import bcolz
from bcolz import carray, ctable
import numpy as np
from pandas import DataFrame
from collections import Iterator
from toolz import partition_all, keyfilter
import os
from datashape import to_numpy_dtype
from toolz import keyfilter
from toolz.curried import pipe, partial, map, concat

from .api.resource import resource
from .dispatch import dispatch
from .compute.bcolz import *
from .utils import keywords


__all__ = ['into', 'bcolz', 'chunks']


@dispatch(type, (ctable, carray))
def into(a, b, **kwargs):
    f = into.dispatch(a, type(b))
    return f(a, b, **kwargs)


@dispatch((tuple, set, list), (ctable, carray))
def into(o, b, **kwargs):
    return into(o, into(np.ndarray(0), b))


@dispatch(Iterator, (ctable, carray))
def into(_, b, **kwargs):
    return pipe(b, chunks, map(partial(into, np.ndarray(0))),
                           map(partial(into, list)),
                           concat)


@dispatch(np.ndarray, (ctable, carray))
def into(a, b, **kwargs):
    return b[:]


@dispatch(ctable, np.ndarray)
def into(a, b, **kwargs):
    return ctable(b, **kwargs)


@dispatch(carray, np.ndarray)
def into(a, b, **kwargs):
    kwargs = keyfilter(keywords(ctable).__contains__, kwargs)
    return carray(b, **kwargs)


@dispatch(carray, (tuple, list))
def into(a, b, dtype=None, **kwargs):
    x = into(np.ndarray(0), b, dtype=dtype)
    kwargs = keyfilter(keywords(ctable).__contains__, kwargs)
    return into(a, x, **kwargs)


@dispatch(carray, carray)
def into(a, b, **kwargs):
    if isinstance(a, type):
        return b
    else:
        a.append(iter(b))
        return a

@dispatch(ctable, (tuple, list))
def into(a, b, names=None, types=None, **kwargs):

    if isinstance(b[0], (tuple, list)):
        if not types:
            types=[None] * len(b[0])
        return ctable([into(np.ndarray(0), c2, dtype=dt)
                        for (c2, dt) in zip(zip(*b), types)], names,
                      **kwargs)
    else:
        if not names:
            names =[None] * len(b)
        arr = into(np.ndarray(0), b, dtype=np.dtype(list(zip(names, types))))
        return ctable(arr, names, **kwargs)


@dispatch((carray, ctable), Iterator)
def into(a, b, **kwargs):
    kwargs = keyfilter(keywords(ctable).__contains__, kwargs)
    chunks = partition_all(1024, b)
    chunk = next(chunks)
    a = into(a, chunk, **kwargs)
    for chunk in chunks:
        a.append(list(zip(*chunk)))
    a.flush()
    return a


@dispatch(DataFrame, ctable)
def into(a, b, columns=None, schema=None, **kwargs):
    if not columns and schema:
        columns = dshape(schema)[0].names
    return DataFrame.from_items(((column, b[column][:]) for column in
                                    sorted(b.names)),
                                orient='columns',
                                columns=columns)


from .compute.chunks import ChunkIterator, chunks

@dispatch((carray, ctable), ChunkIterator)
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


@resource.register('.+\.bcolz/?')
def resource_bcolz(rootdir, **kwargs):
    if os.path.exists(rootdir):
        kwargs = keyfilter(keywords(ctable).__contains__, kwargs)
        return ctable(rootdir=rootdir, **kwargs)
    else:
        if 'dshape' in kwargs:
            dtype = to_numpy_dtype(kwargs['dshape'])
            kwargs = keyfilter(keywords(ctable).__contains__, kwargs)
            return ctable(np.empty(0, dtype), rootdir=rootdir, **kwargs)
        else:
            raise ValueError("File does not exist and no `dshape=` given")
