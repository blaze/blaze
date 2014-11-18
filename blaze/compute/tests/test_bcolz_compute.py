from __future__ import absolute_import, division, print_function

import pytest
from datashape import discover, dshape
bcolz = pytest.importorskip('bcolz')

import numpy as np

import blaze as bz
from blaze.bcolz import into, chunks
from blaze.expr import Symbol
from blaze.compute.core import compute, pre_compute


b = bcolz.ctable(np.array([(1, 1.), (2, 2.), (3, 3.)],
                          dtype=[('a', 'i8'), ('b', 'f8')]))

t = Symbol('t', 'var * {a: int64, b: float64}')


to = Symbol('to', 'var * {a: int64, b: float64}')
bo = bcolz.ctable(np.array([(1, 1.), (2, 2.), (3, np.nan)],
                           dtype=[('a', 'i8'), ('b', 'f8')]))


def test_discover():
    assert discover(b) == dshape('3 * {a: int64, b: float64}')
    assert discover(b['a']) == dshape('3 * int64')


def test_chunks():
    assert len(list(chunks(b, chunksize=2))) == 2
    assert (next(chunks(b, chunksize=2)) == into(np.array(0), b)[:2]).all()


def test_reductions():
    assert compute(t.a.sum(), b) == 6
    assert compute(t.a.min(), b) == 1
    assert compute(t.a.max(), b) == 3
    assert compute(t.a.mean(), b) == 2.0
    assert abs(compute(t.a.std(), b) - np.std([1, 2, 3])) < 1e-5
    assert abs(compute(t.a.var(), b) - np.var([1, 2, 3])) < 1e-5
    assert abs(compute(t.a.std(unbiased=True), b) - np.std([1, 2, 3],
                                                           ddof=1)) < 1e-5
    assert abs(compute(t.a.var(unbiased=True), b) - np.var([1, 2, 3],
                                                           ddof=1)) < 1e-5
    assert len(list(compute(t.distinct(), b))) == 3
    assert len(list(compute(t.a.distinct(), b))) == 3

def test_nunique():
    assert compute(t.a.nunique(), b) == 3
    assert compute(t.nunique(), b) == 3


def test_selection_head():
    b = into(bcolz.ctable,
             ((i, i + 1, float(i)**2) for i in range(10000)),
             names=['a', 'b', 'c'])
    t = Symbol('t', 'var * {a: int32, b: int32, c: float64}')

    assert compute((t.a < t.b).all(), b) == True
    assert list(compute(t[t.a < t.b].a.head(10), b)) == list(range(10))
    assert list(compute(t[t.a > t.b].a.head(10), b)) == []

    assert into([], compute(t[t.a + t.b > t.c], b)) == [(0, 1, 0),
                                                        (1, 2, 1),
                                                        (2, 3, 4)]
    assert len(compute(t[t.a + t.b > t.c].head(10), b)) # non-empty
    assert len(compute(t[t.a + t.b < t.c].head(10), b)) # non-empty


def test_selection_isnan():
    b = bcolz.ctable([[1, np.nan, 3], [1., 2., np.nan]], names=['a', 'b'])
    t = Symbol('t', discover(b))
    lhs = compute(t[t.a.isnan()], b)
    rhs = np.array([(np.nan, 2.0)], dtype=b.dtype)

    for n in b.dtype.names:
        assert np.isclose(lhs[n], rhs[n], equal_nan=True).all()
        assert np.isclose(compute(t[~t.b.isnan()], b)[n],
                          np.array([(1, 1.0), (np.nan, 2.0)], dtype=b.dtype)[n],
                          equal_nan=True).all()


def test_count_isnan():
    assert compute(to.a[~to.b.isnan()].count(), bo) == 2


def test_count_isnan_object():
    assert compute(to.a[~to.b.isnan()].count(), bo) == 2


def test_count_isnan_struct():
    assert compute(t[~t.b.isnan()].count(), b) == 3


def test_nrows():
    assert compute(t.nrows, b) == len(b)


def test_nelements():
    assert compute(t.nelements(axis=0), b) == len(b)
    assert compute(t.nelements(), b) == len(b)


def test_pre_compute():
    b = bcolz.ctable(np.array([(1, 1., 10.), (2, 2., 20.), (3, 3., 30.)],
                              dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'f8')]))

    s = Symbol('s', discover(b))

    result = pre_compute(s[['a', 'b']], b)
    assert result.names == ['a', 'b']
