from blaze.partition import *
from blaze.expr import shape

import numpy as np

x = np.arange(24).reshape(4, 6)


def eq(a, b):
    if isinstance(a == b, bool):
        return a == b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (a == b).all()
    else:
        return a == b


def test_partition_get():
    assert eq(partition_get(x, (0, slice(0, None)), chunksize=(1, 6)),
              x[0, :])
    assert eq(partition_get(x, (slice(0, None), 0), chunksize=(4, 1)),
              x[:, 0])
    assert eq(partition_get(x, (slice(2, 4), slice(0, 2)), chunksize=(2, 2)),
              x[2:4, 0:2])


def test_partition_set():
    x = np.arange(24).reshape(4, 6)
    partition_set(x,
                  (slice(0, 2), slice(0, 2)), np.array([[1, 1], [1, 1]]),
                  chunksize=(2, 2))
    assert (x[:2, :2] == 1).all()


def test_partition_set_1d():
    x = np.arange(24).reshape(4, 6)
    partition_set(x,
                  (slice(0, 4), 0), np.array([[1], [1], [1], [1]]),
                  chunksize=(4, 1),
                  keepdims=False)
    assert (x[:4, 0] == 1).all()


def test_partitions():
    assert list(partitions(x, chunksize=(1, 6))) == \
            [(i, slice(0, 6)) for i in range(4)]
    assert list(partitions(x, chunksize=(4, 1))) == \
            [(slice(0, 4), i) for i in range(6)]
    assert list(partitions(x, chunksize=(2, 3))) == [
            (slice(0, 2), slice(0, 3)), (slice(0, 2), slice(3, 6)),
            (slice(2, 4), slice(0, 3)), (slice(2, 4), slice(3, 6))]


def dont_test_partitions_flat():
    assert list(partitions(x, chunksize=(2, 3))) == [
            (slice(0, 2), slice(0, 3)), (slice(0, 2), slice(3, 6)),
            (slice(2, 4), slice(0, 3)), (slice(2, 4), slice(3, 6))]


def test_uneven_partitions():
    x = np.arange(10*12).reshape(10, 12)
    parts = list(partitions(x, chunksize=(7, 7)))

    assert len(parts) == 2 * 2

    assert parts == [(slice(0,  7), slice(0, 7)), (slice(0,  7), slice(7, 12)),
                     (slice(7, 10), slice(0, 7)), (slice(7, 10), slice(7, 12))]

    x = np.arange(20*24).reshape(20, 24)
    parts = list(partitions(x, chunksize=(7, 7)))


def test_3d_partitions():
    x = np.arange(4*4*6).reshape(4, 4, 6)
    parts = list(partitions(x, chunksize=(2, 2, 3)))
    assert len(parts) == 2 * 2 * 2
