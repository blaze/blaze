from __future__ import absolute_import, division, print_function


import pytest
h5py = pytest.importorskip('h5py')

import sys
from distutils.version import LooseVersion

import numpy as np

from datashape import discover

from blaze import compute
from blaze.expr import symbol
from blaze.utils import tmpfile
from blaze.compute.h5py import pre_compute, optimize


def eq(a, b):
    return (a == b).all()


x = np.arange(20*24, dtype='f4').reshape((20, 24))


@pytest.yield_fixture
def file():
    with tmpfile('.h5') as filename:
        f = h5py.File(filename)
        d = f.create_dataset('/x', shape=x.shape, dtype=x.dtype,
                             fillvalue=0.0, chunks=(4, 6))
        d[:] = x
        yield f
        f.close()


@pytest.yield_fixture
def data(file):
    yield file['/x']


@pytest.yield_fixture
def data_1d_chunks():
    with tmpfile('.h5') as filename:
        f = h5py.File(filename)
        d = f.create_dataset('/x', shape=x.shape, dtype=x.dtype,
                             fillvalue=0.0, chunks=(1, 24))
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

s = symbol('s', discover(x))

two_point_five_and_windows_py3 = \
    pytest.mark.skipif(sys.platform == 'win32' and
                       h5py.__version__ == LooseVersion('2.5.0') and
                       sys.version_info[0] == 3,
                       reason=('h5py 2.5.0 issue with varlen string types: '
                               'https://github.com/h5py/h5py/issues/593'))



def test_slicing(data):
    for idx in [0, 1, (0, 1), slice(1, 3), (0, slice(1, 5, 2))]:
        assert eq(compute(s[idx], data), x[idx])


def test_reductions(data):
    assert eq(compute(s.sum(), data), x.sum())
    assert eq(compute(s.sum(axis=1), data), x.sum(axis=1))
    assert eq(compute(s.sum(axis=0), data), x.sum(axis=0))
    assert eq(compute(s[0], data), x[0])
    assert eq(compute(s[-1], data), x[-1])


@two_point_five_and_windows_py3
def test_mixed(recdata):
    s = symbol('s', discover(recdata))
    expr = (s.x + 1).sum(axis=1)
    assert eq(compute(expr, recdata), compute(expr, rec))


def test_uneven_chunk_size(data):
    assert eq(compute(s.sum(axis=1), data, chunksize=(7, 7)),
              x.sum(axis=1))


@two_point_five_and_windows_py3
def test_nrows_3D_records(recdata):
    s = symbol('s', discover(recdata))
    assert not hasattr(s, 'nrows')


@pytest.mark.xfail(raises=AttributeError,
                   reason="We don't support nrows on arrays")
def test_nrows_array(data):
    assert compute(s.nrows, data) == len(data)


@two_point_five_and_windows_py3
def test_nelements_records(recdata):
    s = symbol('s', discover(recdata))
    assert compute(s.nelements(), recdata) == np.prod(recdata.shape)
    np.testing.assert_array_equal(compute(s.nelements(axis=0), recdata),
                                  np.zeros(recdata.shape[1]) + recdata.shape[0])


def test_nelements_array(data):
    lhs = compute(s.nelements(axis=1), data)
    rhs = data.shape[1]
    np.testing.assert_array_equal(lhs, rhs)

    lhs = compute(s.nelements(axis=0), data)
    rhs = data.shape[0]
    np.testing.assert_array_equal(lhs, rhs)

    lhs = compute(s.nelements(axis=(0, 1)), data)
    rhs = np.prod(data.shape)
    np.testing.assert_array_equal(lhs, rhs)


def test_field_access_on_file(file):
    s = symbol('s', '{x: 20 * 24 * float32}')
    d = compute(s.x, file)
    # assert isinstance(d, h5py.Dataset)
    assert eq(d[:], x)


def test_field_access_on_group(file):
    s = symbol('s', '{x: 20 * 24 * float32}')
    d = compute(s.x, file['/'])
    # assert isinstance(d, h5py.Dataset)
    assert eq(d[:], x)


def test_compute_on_file(file):
    s = symbol('s', discover(file))

    assert eq(compute(s.x.sum(axis=1), file),
              x.sum(axis=1))

    assert eq(compute(s.x.sum(), file, chunksize=(4, 6)),
              x.sum())


def test_compute_on_1d_chunks(data_1d_chunks):
    assert eq(compute(s.sum(), data_1d_chunks),
              x.sum())


def test_arithmetic_on_small_array(data):
    s = symbol('s', discover(data))

    assert eq(compute(s + 1, data),
              compute(s + 1, x))


def test_arithmetic_on_small_array_from_file(file):
    """ Want to make sure that we call pre_compute on Dataset
        Even when it's not the leaf data input. """
    s = symbol('s', discover(file))

    assert eq(compute(s.x + 1, file),
              x + 1)


def test_pre_compute_doesnt_collapse_slices(data):
    s = symbol('s', discover(data))
    assert pre_compute(s[:5], data) is data


def test_optimize_slicing(data):
    a = symbol('a', discover(data))
    b = symbol('b', discover(data))
    assert optimize((a + 1)[:3], data).isidentical(a[:3] + 1)

    assert optimize((a + b)[:3], data).isidentical(a[:3] + b[:3])


def test_optimize_slicing_on_file(file):
    f = symbol('f', discover(file))
    assert optimize((f.x + 1)[:5], file).isidentical(f.x[:5] + 1)


def test_arithmetic_and_then_slicing(data):
    s = symbol('s', discover(data))

    assert eq(compute((2*s + 1)[0], data, pre_compute=False),
              2*x[0] + 1)
