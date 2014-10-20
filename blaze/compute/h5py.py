from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
from multipledispatch import MDNotImplementedError
from datashape import DataShape, to_numpy

from ..partition import partitions, partition_get, partition_set, flatten
from ..expr import Reduction, Field, Projection, Broadcast, Selection, Symbol
from ..expr import Distinct, Sort, Head, Label, ReLabel, Union, Expr, Slice
from ..expr import std, var, count, nunique
from ..expr import BinOp, UnaryOp, USub, Not
from ..expr import path
from ..expr.split import split

from .core import base, compute
from ..dispatch import dispatch
from ..api.into import into

__all__ = []


@dispatch(Slice, h5py.Dataset)
def compute_up(expr, data, **kwargs):
    return data[expr.index]


@dispatch(Expr, h5py.Dataset)
def compute_down(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    if not any(isinstance(node, Reduction) for node in path(expr, leaf)):
        raise MDNotImplementedError()

    # Compute chunksize (this should be improved)
    chunksize = kwargs.get('chunksize', data.chunks)

    # Split expression into per-chunk and on-aggregate pieces
    chunk = Symbol('chunk', DataShape(*(chunksize + (leaf.dshape.measure,))))
    (chunk, chunk_expr), (agg, agg_expr) = \
            split(leaf, expr, chunk=chunk)

    # Create numpy array to hold intermediate aggregate
    shape, dtype = to_numpy(agg.dshape)
    intermediate = np.empty(shape=shape, dtype=dtype)

    # Compute partitions
    data_partitions = partitions(data, blockshape=chunksize)
    int_partitions = partitions(intermediate, blockshape=chunk_expr.shape)

    # For each partition, compute chunk->chunk_expr
    # Insert into intermediate
    for d, i in zip(flatten(data_partitions), flatten(int_partitions)):
        chunk_data = partition_get(data, d, blockshape=chunksize)
        result = compute(chunk_expr, {chunk: chunk_data})
        partition_set(intermediate, i, result, blockshape=chunk_expr.shape)

    # Compute on the aggregate
    return compute(agg_expr, {agg: intermediate})
