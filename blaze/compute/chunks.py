from __future__ import absolute_import, division, print_function

from multipledispatch import MDNotImplementedError
from odo import Chunks, convert, into
from collections import Iterator, Iterable
from toolz import curry, concat
from datashape.dispatch import dispatch

import pandas as pd
import numpy as np

from ..expr import Head, ElemWise, Distinct, Symbol, Expr, path
from ..expr.split import split
from .core import compute
from dask.threaded import get as dsk_get


__all__ = ['Cheap', 'compute_chunk', 'compute_down']


Cheap = (Head, ElemWise, Distinct, Symbol)


@dispatch(Head, Chunks)
def pre_compute(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        return convert(Iterator, data)
    else:
        raise MDNotImplementedError()


def compute_chunk(chunk, chunk_expr, part):
    return compute(chunk_expr, {chunk: part})


@dispatch(Expr, Chunks)
def compute_down(expr, data, **kwargs):

    leaf = expr._leaves()[0]

    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr)
    # Use dask to parallelize the embarassingly parallel per-chunk computation...
    f = curry(compute_chunk, chunk, chunk_expr)
    p = []
    dsk = {}
    for i, d in enumerate(data.data):
        dsk['d%d'%i] = d
        dsk['p%d'%i] = (f, 'd%d'%i)
        p.append('p%d'%i)
    parts = dsk_get(dsk, p)
    # ... then aggregate the result.
    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], (Iterable, Iterator)):
        intermediate = list(concat(parts))

    return compute(agg_expr, {agg: intermediate})


Cheap = (Head, ElemWise, Distinct, Symbol)

@dispatch(Head, Chunks)
def compute_down(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        return compute(expr, {leaf: into(Iterator, data)}, **kwargs)
    else:
        raise MDNotImplementedError()
