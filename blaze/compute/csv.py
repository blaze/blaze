from __future__ import absolute_import, division, print_function

import pandas
import os

from ..dispatch import dispatch
from ..data.csv import CSV
from ..expr import Expr
from ..utils import available_memory
from ..expr.split import split
from toolz import curry, concat
import pandas as pd
import numpy as np
from collections import Iterator, Iterable
from .core import compute

@dispatch(Expr, CSV)
def pre_compute(expr, data, comfortable_memory=None, chunksize=2**18, **kwargs):
    comfortable_memory = comfortable_memory or min(1e9, available_memory() / 4)
    if os.path.getsize(data.path) < comfortable_memory:
        return data.pandas_read_csv()
    else:
        return data.pandas_read_csv(chunksize=chunksize)


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
