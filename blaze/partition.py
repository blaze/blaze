from .dispatch import dispatch

import numpy as np
from math import ceil
import h5py
import toolz

Array = (np.ndarray, h5py.Dataset)

@dispatch(Array, object)
def partition_get(data, part, blockshape=None):
    return data[part]


@dispatch(Array, object, object)
def partition_set(data, part, value, blockshape=None):
    data[part] = value.squeeze()
    return data


def slices1d(n, k):
    """

    >>> slices1d(10, 5)
    [slice(0, 5, None), slice(5, 10, None)]

    >>> slices1d(10, 6)
    [slice(0, 6, None), slice(6, 10, None)]

    >>> slices1d(10, 1)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if k == 1:
        return list(range(n))
    return [slice(k*i, min(k*(i + 1), n)) for i in range(int(ceil(float(n)/k)))]



def tuplepack(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x,)


def slicesnd(shape, blockshape):
    """

    >>> slicesnd((4, 4), (2, 2)) # doctest: +SKIP
    [[(slice(0, 2, None), slice(0, 2, None)),
      (slice(0, 2, None), slice(2, 4, None))]
     [(slice(2, 4, None), slice(0, 2, None)),
      (slice(2, 4, None), slice(2, 4, None))]]
    """
    local = slices1d(shape[0], blockshape[0])
    if len(shape) == 1 and len(blockshape) == 1:
        return local
    else:
        other = slicesnd(shape[1:], blockshape[1:])
        return [[(l,) + tuplepack(o) for o in other]
                                     for l in local]


@dispatch(Array)
def partitions(data, blockshape=None):
    return slicesnd(data.shape, blockshape)


def flatten(x):
    """

    >>> flatten([[1]])
    [1]
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    """
    if isinstance(x[0], list):
        return list(toolz.concat(x))
    else:
        return x
