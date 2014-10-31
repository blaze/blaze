from __future__ import absolute_import, division, print_function

import numpy as np
import h5py

import pytest

from blaze import compute
from blaze.expr import Symbol
from datashape import discover
from blaze.utils import tmpfile

from blaze.compute.h5py import *


def eq(a, b):
    return (a == b).all()


x = np.arange(20*24, dtype='f4').reshape((20, 24))

@pytest.yield_fixture
def data():
    with tmpfile('.h5') as filename:
        f = h5py.File(filename)
        d = f.create_dataset('/x', shape=x.shape, dtype=x.dtype,
                                   fillvalue=0.0, chunks=(4, 6))
        d[:] = x
        yield d
        f.close()


rec = np.empty(shape=(20, 24), dtype=[('x', 'i4'), ('y', 'i4')])
rec['x'] = 1
rec['y'] = 2

@pytest.yield_fixture
def recdata():
    with tmpfile('.h5') as filename:
        f = h5py.File(filename)
        d = f.create_dataset('/x', shape=rec.shape,
                                   dtype=rec.dtype,
                                   chunks=(4, 6))
        d['x'] = rec['x']
        d['y'] = rec['y']
        yield d
        f.close()

s = Symbol('s', discover(x))


def test_slicing(data):
    for idx in [0, 1, (0, 1), slice(1, 3), (0, slice(1, 5, 2))]:
        assert eq(compute(s[idx], data), x[idx])


def test_reductions(data):
    assert eq(compute(s.sum(), data),
              x.sum())
    assert eq(compute(s.sum(axis=1), data),
              x.sum(axis=1))
    assert eq(compute(s.sum(axis=0), data),
              x.sum(axis=0))


def test_mixed(recdata):
    s = Symbol('s', discover(recdata))
    expr = (s.x + 1).sum(axis=1)
    assert eq(compute(expr, recdata), compute(expr, rec))


def test_uneven_chunk_size(data):
    assert eq(compute(s.sum(axis=1), data, chunksize=(7, 7)),
              x.sum(axis=1))


def test_nrows_3D_records(recdata):
    s = Symbol('s', discover(recdata))
    assert not hasattr(s, 'nrows')


def test_nrows_array(data):
    assert compute(s.nrows, data) == len(data)


def test_nelements_records(recdata):
    s = Symbol('s', discover(recdata))
    assert compute(s.nelements(), recdata) == np.prod(recdata.shape)


def test_nelements_array(data):
    assert compute(s.nelements(axis=1), data) == data.shape[1]
