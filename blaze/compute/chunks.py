from __future__ import absolute_import, division, print_function

from into import Chunks, chunks, convert, discover
from collections import Iterator, Iterable
from toolz import curry, concat, map
from datashape.dispatch import dispatch

import pandas as pd
import numpy as np

from ..expr import Head, ElemWise, Distinct, Symbol, Expr, path
from ..expr.split import split
from .core import compute

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
def compute_down(expr, data, map=map, **kwargs):
    leaf = expr._leaves()[0]

    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr)

    parts = map(curry(compute_chunk, chunk, chunk_expr), data)

    if isinstance(parts, Iterator):
        first_part = next(parts)
        parts = concat([[first_part], parts])
    else:
        first_part = parts[0]

    if isinstance(first_part, np.ndarray):
        intermediate = np.concatenate(list(parts))
    elif isinstance(first_part, pd.DataFrame):
        intermediate = pd.concat(list(parts))
    elif isinstance(first_part, (Iterable, Iterator)):
        intermediate = concat(parts)

    return compute(agg_expr, {agg: intermediate})
