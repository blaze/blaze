from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
from multipledispatch import MDNotImplementedError
from operator import mul

from ..expr import *
from ..expr.split import *

from .core import base, compute
from ..dispatch import dispatch
from ..api.into import into
from ..partition import partitions, partition_get, partition_set

__all__ = []


@dispatch(Slice, h5py.Dataset)
def compute_up(expr, data, **kwargs):
    return data[expr.index]


def listshape(L):
    """

    >>> listshape([1, 2, 3])
    (3,)
    >>> listshape([[1, 2], [3, 4], [5, 6]])
    (3, 2)
    """
    result = ()
    while isinstance(L, list):
        result = result + (len(L),)
        L = L[0]
    return result


@dispatch(Expr, h5py.Dataset)
def compute_down(expr, data, **kwargs):
    chunkshape = data.chunks
    assert blockshape

    leaf = expr._leaves()[0]
    chunk = Symbol('chunk', DataShape(*(chunkshape + (leaf.dshape.measure,))))
    try:
        (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr, chunk=chunk)
    except:
        raise MDNotImplementedError()


    leaf_partitions = partitions(data, blockshape=blockshape)

    chunkshape, dtype = datashape.to_numpy(chunk_expr.dshape)

    intermediate = np.empty(shape=tuple(map(mul, shape, listshape(leaf_partitions)),
                            dtype=dtype))

    intermediate_partitions = partitions(intermediate, blockshape=shape)

    for source, target in zip(flatten(leaf_partitions),
                              flatten(intermediate_partitions)):
        inp = partition_get(data, source, blockshape=blockshape)
        out = compute(chunk_expr, {chunk: inp})
        partition_set(intermediate, target, out, blockshape=shape)

    result = compute(agg_expr, {agg: intermediate})
    return result
