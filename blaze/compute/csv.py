from __future__ import absolute_import, division, print_function

import pandas
import os
from toolz import curry, concat
import pandas as pd
import numpy as np
from collections import Iterator, Iterable
from odo import into, Temp
from odo.chunks import chunks
from odo.backends.csv import CSV
from odo.backends.url import URL
from multipledispatch import MDNotImplementedError

from ..dispatch import dispatch
from ..expr import Expr, Head, ElemWise, Distinct, Symbol, Projection, Field
from ..expr.core import path
from ..utils import available_memory
from ..expr.split import split
from .core import compute
from ..expr.optimize import lean_projection
from .pmap import get_default_pmap


__all__ = ['optimize', 'pre_compute', 'compute_chunk', 'compute_down']


@dispatch(Expr, CSV)
def optimize(expr, _):
    return lean_projection(expr)  # This is handled in pre_compute


@dispatch(Expr, CSV)
def pre_compute(expr, data, comfortable_memory=None, chunksize=2**18, **kwargs):
    comfortable_memory = comfortable_memory or min(1e9, available_memory() / 4)

    kwargs = dict()

    # Chunk if the file is large
    if os.path.getsize(data.path) > comfortable_memory:
        kwargs['chunksize'] = chunksize
    else:
        chunksize = None

    # Insert projection into read_csv
    oexpr = optimize(expr, data)
    leaf = oexpr._leaves()[0]
    pth = list(path(oexpr, leaf))
    if len(pth) >= 2 and isinstance(pth[-2], (Projection, Field)):
        # NOTE: FIXME: We pass the column names through `str` to workaround a
        # PY2 Pandas bug with strings / unicode objects.
        kwargs['usecols'] = list(map(str, pth[-2].fields))

    if chunksize:
        return into(chunks(pd.DataFrame), data, dshape=leaf.dshape, **kwargs)
    else:
        return into(pd.DataFrame, data, dshape=leaf.dshape, **kwargs)


@dispatch((Expr, Head), URL(CSV))
def pre_compute(expr, data, **kwargs):
    return pre_compute(expr, into(Temp(CSV), data, **kwargs), **kwargs)


Cheap = (Head, ElemWise, Distinct, Symbol)

@dispatch(Head, CSV)
def pre_compute(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        return into(Iterator, data, chunksize=10000, dshape=leaf.dshape)
    else:
        raise MDNotImplementedError()


def compute_chunk(chunk, chunk_expr, part):
    return compute(chunk_expr, {chunk: part}, return_type='native')


@dispatch(Expr, pandas.io.parsers.TextFileReader)
def compute_down(expr, data, map=None, **kwargs):
    if map is None:
        map = get_default_pmap()
    leaf = expr._leaves()[0]

    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr)

    parts = list(map(curry(compute_chunk, chunk, chunk_expr), data))

    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], (Iterable, Iterator)):
        intermediate = concat(parts)

    return compute(agg_expr, {agg: intermediate}, return_type='native')
