from __future__ import absolute_import, division, print_function

from toolz import curry, concat, first
from multipledispatch import MDNotImplementedError

from ..expr import (Selection, Head, Field, Projection, ReLabel, ElemWise,
                    Arithmetic, Broadcast, Symbol, Summary, Like, Sort, Apply,
                    Reduction, symbol, IsIn)
from ..expr import Label, Distinct, By, Slice
from ..expr import Expr
from ..expr import path
from ..expr.optimize import lean_projection
from ..expr.split import split
from ..partition import partitions
from .core import compute
from .pmap import get_default_pmap

from collections import Iterator, Iterable
import datashape
import bcolz
import numpy as np
import pandas as pd


from ..dispatch import dispatch
from odo import into

__all__ = ['bcolz']

COMFORTABLE_MEMORY_SIZE = 1e9


@dispatch(Expr, (bcolz.ctable, bcolz.carray))
def optimize(expr, _):
    return lean_projection(expr)  # This is handled in pre_compute


@dispatch(Expr, (bcolz.ctable, bcolz.carray))
def pre_compute(expr, data, scope=None, **kwargs):
    return data


@dispatch((bcolz.carray, bcolz.ctable))
def discover(data):
    return datashape.from_numpy(data.shape, data.dtype)

Cheap = (Head, ElemWise, Distinct, Symbol)


@dispatch(Head, (bcolz.ctable, bcolz.carray))
def compute_down(expr, data, **kwargs):
    """ Cheap and simple computation in simple case

    If we're given a head and the entire expression is cheap to do (e.g.
    elemwises, selections, ...) then compute on data directly, without
    parallelism"""
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        return compute(expr, {leaf: into(Iterator, data)}, **kwargs)
    else:
        raise MDNotImplementedError()


@dispatch((Broadcast, Arithmetic, ReLabel, Summary, Like, Sort, Label, Head,
           Selection, ElemWise, Apply, Reduction, Distinct, By, IsIn),
          (bcolz.ctable, bcolz.carray))
def compute_up(expr, data, **kwargs):
    """ This is only necessary because issubclass(bcolz.carray, Iterator)

    So we have to explicitly avoid the streaming Python backend"""
    raise NotImplementedError()


@dispatch(Field, bcolz.ctable)
def compute_up(expr, data, **kwargs):
    return data[str(expr._name)]


@dispatch(Projection, bcolz.ctable)
def compute_up(expr, data, **kwargs):
    return data[list(map(str, expr.fields))]


@dispatch(Slice, (bcolz.carray, bcolz.ctable))
def compute_up(expr, x, **kwargs):
    return x[expr.index]


def compute_chunk(source, chunk, chunk_expr, data_index):
    part = source[data_index]
    return compute(chunk_expr, {chunk: part})


def get_chunksize(data):
    if isinstance(data, bcolz.carray):
        return data.chunklen
    elif isinstance(data, bcolz.ctable):
        return min(data[c].chunklen for c in data.names)
    else:
        raise TypeError("Don't know how to compute chunksize for type %r" %
                        type(data).__name__)


@dispatch(Expr, (bcolz.carray, bcolz.ctable))
def compute_down(expr, data, chunksize=None, map=None, **kwargs):
    if map is None:
        map = get_default_pmap()

    leaf = expr._leaves()[0]

    if chunksize is None:
        chunksize = max(2**16, get_chunksize(data))

    # If the bottom expression is a projection or field then want to do
    # compute_up first
    children = set(e for e in expr._traverse()
                   if isinstance(e, Expr)
                   and any(i is expr._leaves()[0] for i in e._inputs))
    if len(children) == 1 and isinstance(first(children), (Field, Projection)):
        raise MDNotImplementedError()

    chunk = symbol('chunk', chunksize * leaf.schema)
    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr, chunk=chunk)

    data_parts = partitions(data, chunksize=(chunksize,))

    parts = list(map(curry(compute_chunk, data, chunk, chunk_expr),
                     data_parts))

    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], Iterable):
        intermediate = list(concat(parts))
    else:
        raise TypeError("Don't know how to concatenate objects of type %r" %
                        type(parts[0]).__name__)

    return compute(agg_expr, {agg: intermediate})
