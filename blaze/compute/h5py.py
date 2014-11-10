from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
from multipledispatch import MDNotImplementedError
from datashape import DataShape, to_numpy

from ..partition import partitions, partition_get, partition_set, flatten
from ..expr import Reduction, Field, Projection, Broadcast, Selection, Symbol
from ..expr import Distinct, Sort, Head, Label, ReLabel, Union, Expr, Slice
from ..expr import std, var, count, nunique
from ..expr import BinOp, UnaryOp, USub, Not, nelements
from ..expr import path
from ..expr.split import split

from .core import base, compute
from ..dispatch import dispatch
from ..api.into import into
from ..partition import partitions, partition_get, partition_set

__all__ = []


@dispatch(Symbol, (h5py.File, h5py.Group, h5py.Dataset))
def compute_up(expr, data, **kwargs):
    return data


@dispatch(Field, (h5py.File, h5py.Group))
def compute_up(expr, data, **kwargs):
    return data[expr._name]


@dispatch(Slice, h5py.Dataset)
def compute_up(expr, data, **kwargs):
    return data[expr.index]


@dispatch(nelements, h5py.Dataset)
def compute_up(expr, data, **kwargs):
    return compute_up.dispatch(type(expr), np.ndarray)(expr, data, **kwargs)


@dispatch(Expr, (h5py.File, h5py.Group))
def compute_down(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    p = list(path(expr, leaf))[::-1][1:]
    if not p:
        return data
    for e in p:
        data = compute_up(e, data)
        if not isinstance(data, (h5py.File, h5py.Group)):
            break

    expr2 = expr._subs({e: Symbol('leaf', e.dshape)})
    return compute_down(expr2, data, **kwargs)


@dispatch(Expr, h5py.Dataset)
def compute_down(expr, data, **kwargs):
    """ Compute expressions on H5Py datasets by operating on chunks

    This uses blaze.expr.split to break a full-array-computation into a
    per-chunk computation and a on-aggregate computation.

    This uses blaze.partition to pick out chunks from the h5py dataset, uses
    compute(numpy) to compute on each chunk and then uses blaze.partition to
    aggregate these (hopefully smaller) intermediate results into a local
    numpy array.  It then performs a second operation (again given by
    blaze.expr.split) on this intermediate aggregate

    The expression must contain some sort of Reduction.  Both the intermediate
    result and the final result are assumed to fit into memory
    """
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
    data_partitions = partitions(data, chunksize=chunksize)
    int_partitions = partitions(intermediate, chunksize=chunk_expr.shape)

    # For each partition, compute chunk->chunk_expr
    # Insert into intermediate
    # This could be parallelized
    for d, i in zip(data_partitions, int_partitions):
        chunk_data = partition_get(data, d, chunksize=chunksize)
        result = compute(chunk_expr, {chunk: chunk_data})
        partition_set(intermediate, i, result, chunksize=chunk_expr.shape)

    # Compute on the aggregate
    return compute(agg_expr, {agg: intermediate})
