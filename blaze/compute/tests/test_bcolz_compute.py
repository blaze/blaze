from __future__ import absolute_import, division, print_function

import pytest
bcolz = pytest.importorskip('bcolz')

import numpy as np

from blaze.bcolz import into, chunks
from blaze.expr import TableSymbol
from blaze.compute.core import compute


b = bcolz.ctable([[1, 2, 3], [1., 2., 3.]],
                 names=['a', 'b'])


t = TableSymbol('t', '{a: int32, b: float64}')


to = TableSymbol('to', '{a: int32, b: float64}')
bo = bcolz.ctable([[1, 2, 3], [1., 2., np.nan]],
                  names=['a', 'b'])

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
    assert compute(t.a.nunique(), b) == 3
    assert compute(t.nunique(), b) == 3
    assert len(list(compute(t.distinct(), b))) == 3
    assert len(list(compute(t.a.distinct(), b))) == 3

    assert compute(t.a.count(), b) == 3
    assert compute(t.b.count(), b) == 3


def test_count_nulls():
    c = bcolz.ctable([[1.0, 2.0, np.nan], [1., np.nan, np.nan]],
                     names=list('ab'))
    ct = TableSymbol('t', '{a: float64, b: float64}')
    lhs = compute(ct.count(), c)
    rhs = np.array([(2, 1)], dtype=[('a', 'int64'), ('b', 'int64')])
    rhs = np.array([(2, 1)], dtype=[('a', 'int64'), ('b', 'int64')])
    assert (lhs == rhs).all()

    assert compute(ct.a.count(), c) == 2
    assert compute(ct.b.count(), c) == 1


def test_selection_head():
    b = into(bcolz.ctable,
             ((i, i + 1, float(i)**2) for i in range(10000)),
             names=['a', 'b', 'c'])
    t = TableSymbol('t', '{a: int32, b: int32, c: float64}')

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
    lhs = compute(t[t.a.isnan()], b)
    rhs = np.array([(np.nan, 2.0)], dtype=b.dtype)

    for n in b.dtype.names:
        assert np.isclose(lhs[n], rhs[n], equal_nan=True).all()
        assert np.isclose(compute(t[~t.b.isnan()], b)[n],
                          np.array([(1, 1.0), (np.nan, 2.0)], dtype=b.dtype)[n],
                          equal_nan=True).all()


def test_count_isnan():
    assert compute(to.b[to.a.isnan()].count(), bo) == 0
    assert compute(to.a[~to.b.isnan()].count(), bo) == 2


def test_count_isnan_object():
    assert compute(to.a[~to.b.isnan()].count(), bo) == 2


@pytest.mark.xfail(raises=TypeError,
                   reason="isnan doesn't work on struct/record dtypes")
def test_count_isnan_struct():
    assert compute(t[~t.a.isnan()].count(), b) == 2  # 3?
