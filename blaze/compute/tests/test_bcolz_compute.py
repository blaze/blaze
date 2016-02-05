from __future__ import absolute_import, division, print_function

import pytest
bcolz = pytest.importorskip('bcolz')

from datashape import discover, dshape

import numpy as np

import pandas.util.testing as tm

from odo import into
from blaze import by
from blaze.expr import symbol
from blaze.compute.core import compute, pre_compute
from blaze.compute.bcolz import get_chunksize


b = bcolz.ctable(np.array([(1, 1., np.datetime64('2010-01-01')),
                           (2, 2., np.datetime64('NaT')),
                           (3, 3., np.datetime64('2010-01-03'))],
                          dtype=[('a', 'i8'),
                                 ('b', 'f8'),
                                 ('date', 'datetime64[D]')]))

t = symbol('t', 'var * {a: int64, b: float64, date: ?date}')

to = symbol('to', 'var * {a: int64, b: float64}')
bo = bcolz.ctable(np.array([(1, 1.), (2, 2.), (3, np.nan)],
                           dtype=[('a', 'i8'), ('b', 'f8')]))


def test_discover():
    assert discover(b) == dshape('3 * {a: int64, b: float64, date: date}')
    assert discover(b['a']) == dshape('3 * int64')


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

    assert compute(t.a.nunique(), b) == 3
    assert isinstance(compute(t.a.nunique(), b), np.integer)

    assert compute(t.a.count(), b) == 3
    assert isinstance(compute(t.date.count(), b), np.integer)

    assert compute(t.date.nunique(), b) == 2
    assert isinstance(compute(t.date.nunique(), b), np.integer)

    assert compute(t.date.count(), b) == 2
    assert isinstance(compute(t.a.count(), b), np.integer)

    assert compute(t.a[0], b) == 1
    assert compute(t.a[-1], b) == 3
    assert compute(t[0], b) == compute(t[0], b)
    assert compute(t[-1], b) == compute(t[-1], b)


def test_nunique():
    assert compute(t.a.nunique(), b) == 3
    assert compute(t.nunique(), b) == 3


def test_selection_head():
    ds = dshape('var * {a: int32, b: int32, c: float64}')
    b = into(bcolz.ctable,
             [(i, i + 1, float(i) ** 2) for i in range(10)],
             dshape=ds)
    t = symbol('t', ds)

    # numpy reductions return numpy scalars
    assert compute((t.a < t.b).all(), b).item() is True
    assert list(compute(t[t.a < t.b].a.head(10), b)) == list(range(10))
    assert list(compute(t[t.a > t.b].a.head(10), b)) == []

    assert into([], compute(t[t.a + t.b > t.c], b)) == [(0, 1, 0),
                                                        (1, 2, 1),
                                                        (2, 3, 4)]
    assert len(compute(t[t.a + t.b > t.c].head(10), b))  # non-empty
    assert len(compute(t[t.a + t.b < t.c].head(10), b))  # non-empty


def test_selection_isnan():
    b = bcolz.ctable([[1, np.nan, 3], [1., 2., np.nan]], names=['a', 'b'])
    t = symbol('t', discover(b))
    lhs = compute(t[t.a.isnan()], b)
    rhs = np.array([(np.nan, 2.0)], dtype=b.dtype)

    for n in b.dtype.names:
        assert np.isclose(lhs[n], rhs[n], equal_nan=True).all()
        assert np.isclose(compute(t[~t.b.isnan()], b)[n],
                          np.array(
                              [(1, 1.0), (np.nan, 2.0)], dtype=b.dtype)[n],
                          equal_nan=True).all()


def test_count_isnan():
    assert compute(to.a[~to.b.isnan()].count(), bo) == 2


def test_count_isnan_struct():
    assert compute(t[~t.b.isnan()].count(), b) == 3


def test_nrows():
    assert compute(t.nrows, b) == len(b)


def test_nelements():
    assert compute(t.nelements(axis=0), b) == len(b)
    assert compute(t.nelements(), b) == len(b)


# This is no longer desired. Handled by compute_up
def dont_test_pre_compute():
    b = bcolz.ctable(np.array([(1, 1., 10.), (2, 2., 20.), (3, 3., 30.)],
                              dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'f8')]))

    s = symbol('s', discover(b))

    result = pre_compute(s[['a', 'b']], b)
    assert result.names == ['a', 'b']


def eq(a, b):
    return np.array_equal(a, b)


def test_unicode_field_names():
    b = bcolz.ctable(np.array([(1, 1., 10.), (2, 2., 20.), (3, 3., 30.)],
                              dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'f8')]))
    s = symbol('s', discover(b))

    assert eq(compute(s[u'a'], b)[:], compute(s['a'], b)[:])
    assert eq(compute(s[[u'a', u'c']], b)[:], compute(s[['a', 'c']], b)[:])
    assert eq(compute(s[u'a'], b)[:],
              compute(s['a'],  b)[:])
    assert eq(compute(s[[u'a', u'c']], b)[:],
              compute(s[['a', 'c']],  b)[:])


def test_chunksize_inference():
    b = bcolz.ctable(np.array([(1, 1., 10.), (2, 2., 20.), (3, 3., 30.)],
                              dtype=[('a', 'i8'), ('b', 'f8'), ('c', 'f8')]),
                     chunklen=2)
    assert get_chunksize(b) == 2


def test_notnull():
    with pytest.raises(AttributeError):
        t.b.notnull


def test_by_with_single_row():
    ct = bcolz.ctable([[1, 1, 3, 3], [1, 2, 3, 4]], names=list('ab'))
    t = symbol('t', discover(ct))
    subset = t[t.a == 3]
    expr = by(subset.a, b_sum=subset.b.sum())
    result = compute(expr, ct)
    expected = compute(expr, ct, optimize=False)
    tm.assert_frame_equal(result, expected)
