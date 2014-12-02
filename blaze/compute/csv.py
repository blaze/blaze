from __future__ import absolute_import, division, print_function

import pandas
import os

from ..dispatch import dispatch
from ..data.csv import CSV
from ..expr import Expr, Head, ElemWise, Distinct, Symbol, Projection, Field
from ..expr.core import path
from ..utils import available_memory
from ..expr.split import split
from ..api.into import into
from toolz import curry, concat, map
import pandas as pd
import numpy as np
from collections import Iterator, Iterable
from .core import compute

@dispatch(Expr, CSV)
def pre_compute(expr, data, comfortable_memory=None, chunksize=2**18, **kwargs):
    comfortable_memory = comfortable_memory or min(1e9, available_memory() / 4)

    kwargs = dict()

    # Chunk if the file is large
    if os.path.getsize(data.path) > comfortable_memory:
        kwargs['chunksize'] = chunksize

    # Insert projection into read_csv
    leaf = expr._leaves()[0]
    pth = list(path(expr, leaf))
    if len(pth) >= 2 and isinstance(pth[-2], (Projection, Field)):
        kwargs['usecols'] = pth[-2].fields

    return data.pandas_read_csv(**kwargs)


Cheap = (Head, ElemWise, Distinct, Symbol)

@dispatch(Head, CSV)
def pre_compute(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        return into(Iterator, data)
    else:
        raise MDNotImplementedError()


def compute_chunk(chunk, chunk_expr, part):
    return compute(chunk_expr, {chunk: part})


@dispatch(Expr, pandas.io.parsers.TextFileReader)
def compute_down(expr, data, map=map, **kwargs):
    leaf = expr._leaves()[0]

    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr)

    parts = list(map(curry(compute_chunk, chunk, chunk_expr), data))

    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], (Iterable, Iterator)):
        intermediate = concat(parts)

    return compute(agg_expr, {agg: intermediate})


@dispatch(Iterator, pandas.io.parsers.TextFileReader)
def into(a, b, **kwargs):
    return concat(map(into(a), b))


@dispatch((list, tuple, set), pandas.io.parsers.TextFileReader)
def into(a, b, **kwargs):
    return into(a, into(Iterator, b))
