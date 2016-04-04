from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
from multipledispatch import MDNotImplementedError
from datashape import DataShape, to_numpy
from toolz import curry

import psutil

from multiprocessing.pool import ThreadPool

from ..partition import partitions
from ..expr import Reduction, Field, symbol
from ..expr import Expr, Slice, ElemWise
from ..expr import nelements
from ..expr import path, shape, Symbol
from ..expr.split import split

from .core import compute, compute_single_object
from ..dispatch import dispatch
from ..utils import available_memory


__all__ = []


# h5py File and Group are mapping objects.
compute.register(Expr, (h5py.File, h5py.Group))(compute_single_object)


@dispatch(Slice, h5py.Dataset)
def pre_compute(expr, data, scope=None, **kwargs):
    """ Don't push slices into memory, they're about to come in anyway """
    return data


@dispatch(Expr, h5py.Dataset)
def pre_compute(expr, data, scope=None, **kwargs):
    """ Bring dataset into memory if it's small relative to memory """
    nbytes = data.size * data.dtype.alignment
    comfortable_memory = available_memory() / 4

    if nbytes < comfortable_memory:
        return data.value
    else:
        return data


@dispatch(Expr, h5py.Dataset)
def post_compute(expr, data, scope=None):
    """ Bring dataset into memory if it's small relative to memory """
    nbytes = data.size * data.dtype.alignment
    comfortable_memory = available_memory() / 4

    if nbytes < comfortable_memory:
        return data.value
    else:
        return data


@dispatch(Symbol, h5py.Dataset)
def optimize(expr, data):
    return expr


@dispatch(Expr, h5py.Dataset)
def optimize(expr, data):
    child = optimize(expr._inputs[0], data)
    if child is expr._inputs[0]:
        return expr
    else:
        return expr._subs({expr._inputs[0]: child})


@dispatch(Slice, (h5py.File, h5py.Group, h5py.Dataset))
def optimize(expr, data):
    child = expr._inputs[0]
    if (isinstance(child, ElemWise) and len(child._inputs) == 1
            and shape(child._inputs[0]) == shape(child)):
        grandchild = child._inputs[0][expr.index]
        grandchild = optimize(grandchild, data)
        return child._subs({child._inputs[0]: grandchild})
    if (isinstance(child, ElemWise) and len(child._inputs) == 2
        and shape(child) == shape(expr._inputs[0]) == shape(child._inputs[1])):
        lhs, rhs = child._inputs
        lhs = lhs[expr.index]
        rhs = rhs[expr.index]
        lhs = optimize(lhs, data)
        rhs = optimize(rhs, data)
        return child._subs(dict(zip(child._inputs, (lhs, rhs))))
    else:
        return expr


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

    expr2 = expr._subs({e: symbol('leaf', e.dshape)})
    return compute_down(expr2, data, **kwargs)


def compute_chunk(source, target, chunk, chunk_expr, parts):
    """ Pull out a part, compute it, insert it into the target """
    source_part, target_part = parts
    part = source[source_part]
    result = compute(chunk_expr, {chunk: part}, return_type='native')
    target[target_part] = result


thread_pool = None

def _get_map(map):
    global thread_pool

    if map is None:
        if thread_pool is None:
            thread_pool = ThreadPool(psutil.cpu_count())
        return thread_pool.map
    else:
        return map


@dispatch(Expr, h5py.Dataset)
def compute_down(expr, data, map=None, **kwargs):
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
    map = _get_map(map)

    leaf = expr._leaves()[0]
    if not any(isinstance(node, Reduction) for node in path(expr, leaf)):
        raise MDNotImplementedError()

    # Compute chunksize (this should be improved)
    chunksize = kwargs.get('chunksize', data.chunks)

    # Split expression into per-chunk and on-aggregate pieces
    chunk = symbol('chunk', DataShape(*(chunksize + (leaf.dshape.measure,))))
    (chunk, chunk_expr), (agg, agg_expr) = \
            split(leaf, expr, chunk=chunk)

    # Create numpy array to hold intermediate aggregate
    shape, dtype = to_numpy(agg.dshape)
    intermediate = np.empty(shape=shape, dtype=dtype)

    # Compute partitions
    source_parts = list(partitions(data, chunksize=chunksize, keepdims=True))
    target_parts = list(partitions(intermediate, chunksize=chunk_expr.shape,
                                   keepdims=True))

    list(map(
        curry(compute_chunk, data, intermediate, chunk, chunk_expr),
        zip(source_parts, target_parts)
    ))

    # Compute on the aggregate
    return compute(agg_expr, {agg: intermediate}, return_type='native')
