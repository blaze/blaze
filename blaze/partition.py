""" Abstraction around partitionable data

Some data can be partitioned into disjoint pieces.
Examples include lists, tuples, and arrays of various flavors.

This module provides functions to abstract away the handling of such
partitions.  It proposes three operations, each of these operations deals with
data (like a numpy array) and a *partition-key* which is some token to define a
part of the data.  In the case of numpy (and other array-like data) a natural
partition-key might be a tuple of slices, e.g.

    (slice(0, 10), slice(0, 10)) :: partition-key

Our three operations:


    partitions :: data -> [partition-keys]
        Get list or list-of-lists of partition-keys

    partition_get :: data, partition-key -> partition
        Get a particular partition from a dataset

    partition_set :: data, partition-key, value -> void
        Set the value of a particular partition in a dataset

Using these three operations we should be able to write down very simple and
abstract algorithms like "copy"

    def copy(in_data, out_data):
        for in_key, out_key in zip(partitions(in_data), partitions(out_data)):
            data = partition_get(in_data, in_key)
            partition_set(out_data, out_key, data)
"""
from .dispatch import dispatch

import numpy as np
from math import ceil
import h5py
import toolz
import itertools

Array = (np.ndarray, h5py.Dataset)

@dispatch(Array, object)
def partition_get(data, part, chunksize=None):
    return data[part]


@dispatch(Array, object, object)
def partition_set(data, part, value, chunksize=None):
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
    if isinstance(x, (tuple, list)):
        return tuple(x)
    else:
        return (x,)


@dispatch(Array)
def partitions(data, chunksize=None):
    per_dim = map(slices1d, data.shape, chunksize)
    return itertools.product(*per_dim)


def flatten(x):
    """

    >>> flatten([[1]])
    [1]
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    >>> flatten([[[1], [2]], [[3], [4]]])
    [1, 2, 3, 4]
    """
    if isinstance(x[0], list):
        return list(toolz.concat(map(flatten, x)))
    else:
        return x
