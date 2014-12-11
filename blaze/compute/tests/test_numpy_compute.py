from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
from datetime import datetime, date

from blaze.compute.core import compute, compute_up
from blaze.expr import symbol, by, exp, summary
from into import into
from datashape import discover, to_numpy


t = symbol('t', 'var * {id: int, name: string, amount: int}')

x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
            dtype=[('id', 'i8'), ('name', 'S7'), ('amount', 'i8')])

def eq(a, b):
    c = a == b
    if isinstance(c, np.ndarray):
        return c.all()
    return c


def test_symbol():
    assert eq(compute(t, x), x)


def test_eq():
    assert eq(compute(t['amount'] == 100, x),
              x['amount'] == 100)


def test_selection():
    assert eq(compute(t[t['amount'] == 100], x), x[x['amount'] == 0])
    assert eq(compute(t[t['amount'] < 0], x), x[x['amount'] < 0])


def test_arithmetic():
    assert eq(compute(t['amount'] + t['id'], x),
              x['amount'] + x['id'])
    assert eq(compute(t['amount'] * t['id'], x),
              x['amount'] * x['id'])
    assert eq(compute(t['amount'] % t['id'], x),
              x['amount'] % x['id'])


def test_UnaryOp():
    assert eq(compute(exp(t['amount']), x),
              np.exp(x['amount']))


def test_Neg():
    assert eq(compute(-t['amount'], x),
              -x['amount'])


def test_invert_not():
    assert eq(compute(~(t.amount > 0), x),
              ~(x['amount'] > 0))


def test_Reductions():
    assert compute(t['amount'].mean(), x) == x['amount'].mean()
    assert compute(t['amount'].count(), x) == len(x['amount'])
    assert compute(t['amount'].sum(), x) == x['amount'].sum()
    assert compute(t['amount'].min(), x) == x['amount'].min()
    assert compute(t['amount'].max(), x) == x['amount'].max()
    assert compute(t['amount'].nunique(), x) == len(np.unique(x['amount']))
    assert compute(t['amount'].var(), x) == x['amount'].var()
    assert compute(t['amount'].std(), x) == x['amount'].std()
    assert compute(t['amount'].var(unbiased=True), x) == x['amount'].var(ddof=1)
    assert compute(t['amount'].std(unbiased=True), x) == x['amount'].std(ddof=1)
    assert compute((t['amount'] > 150).any(), x) == True
    assert compute((t['amount'] > 250).all(), x) == False

def test_reductions_on_recarray():
    assert compute(t.count(), x) == len(x)

def test_count_nan():
    t = symbol('t', '3 * ?real')
    x = np.array([1.0, np.nan, 2.0])
    assert compute(t.count(), x) == 2


def test_Distinct():
    x = np.array([('Alice', 100),
                  ('Alice', -200),
                  ('Bob', 100),
                  ('Bob', 100)],
                dtype=[('name', 'S5'), ('amount', 'i8')])

    t = symbol('t', 'var * {name: string, amount: int64}')

    assert eq(compute(t['name'].distinct(), x),
              np.unique(x['name']))
    assert eq(compute(t.distinct(), x),
              np.unique(x))


def test_sort():
    assert eq(compute(t.sort('amount'), x),
              np.sort(x, order='amount'))

    assert eq(compute(t.sort('amount', ascending=False), x),
              np.sort(x, order='amount')[::-1])

    assert eq(compute(t.sort(['amount', 'id']), x),
              np.sort(x, order=['amount', 'id']))


def test_head():
    assert eq(compute(t.head(2), x),
              x[:2])


def test_label():
    expected = x['amount'] * 10
    expected = np.array(expected, dtype=[('foo', 'i8')])
    assert eq(compute((t['amount'] * 10).label('foo'), x),
              expected)


def test_relabel():
    expected = np.array(x, dtype=[('ID', 'i8'), ('NAME', 'S7'), ('amount', 'i8')])
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), x)

    assert result.dtype.names == expected.dtype.names
    assert eq(result, expected)


def test_by():
    expr = by(t.amount > 0, count=t.id.count())
    result = compute(expr, x)
    assert set(map(tuple, into(list, result))) == set([(False, 2), (True, 3)])


def test_compute_up_field():
    assert eq(compute(t['name'], x), x['name'])


def test_compute_up_projection():
    assert eq(compute_up(t[['name', 'amount']], x), x[['name', 'amount']])


def test_slice():
    for s in [0, slice(2), slice(1, 3), slice(None, None, 2)]:
        assert (compute(t[s], x) == x[s]).all()



ax = np.arange(30, dtype='f4').reshape((5, 3, 2))

a = symbol('a', discover(ax))

def test_array_reductions():
    for axis in [None, 0, 1, (0, 1), (2, 1)]:
        assert eq(compute(a.sum(axis=axis), ax), ax.sum(axis=axis))


def test_array_reductions_with_keepdims():
    for axis in [None, 0, 1, (0, 1), (2, 1)]:
        assert eq(compute(a.sum(axis=axis, keepdims=True), ax),
                 ax.sum(axis=axis, keepdims=True))


def test_summary_on_ndarray():
    assert compute(summary(total=a.sum(), min=a.min()), ax) == \
            (ax.min(), ax.sum())

    result = compute(summary(total=a.sum(), min=a.min(), keepdims=True), ax)
    expected = np.array([(ax.min(), ax.sum())],
                        dtype=[('min', 'f4'), ('total', 'f4')])
    assert result.ndim == ax.ndim
    assert eq(expected, result)


def test_summary_on_ndarray_with_axis():
    for axis in [0, 1, (1, 0)]:
        expr = summary(total=a.sum(), min=a.min(), axis=axis)
        result = compute(expr, ax)

        shape, dtype = to_numpy(expr.dshape)
        expected = np.empty(shape=shape, dtype=dtype)
        expected['total'] = ax.sum(axis=axis)
        expected['min'] = ax.min(axis=axis)

        assert eq(result, expected)


def test_utcfromtimestamp():
    t = symbol('t', '1 * int64')
    data = np.array([0, 1])
    expected = np.array(['1970-01-01T00:00:00Z', '1970-01-01T00:00:01Z'],
                        dtype='M8[us]')
    assert eq(compute(t.utcfromtimestamp, data), expected)


def test_nelements_structured_array():
    assert compute(t.nelements(), x) == len(x)
    assert compute(t.nelements(keepdims=True), x) == (len(x),)


def test_nelements_array():
    t = symbol('t', '5 * 4 * 3 * float64')
    x = np.random.randn(*t.shape)
    result = compute(t.nelements(axis=(0, 1)), x)
    np.testing.assert_array_equal(result, np.array([20, 20, 20]))

    result = compute(t.nelements(axis=1), x)
    np.testing.assert_array_equal(result, 4 * np.ones((5, 3)))


def test_nrows():
    assert compute(t.nrows, x) == len(x)


dts = np.array(['2000-06-25T12:30:04Z', '2000-06-28T12:50:05Z'],
               dtype='M8[us]')
s = symbol('s', 'var * datetime')

def test_datetime_truncation():

    assert eq(compute(s.truncate(1, 'day'), dts),
              dts.astype('M8[D]'))
    assert eq(compute(s.truncate(2, 'seconds'), dts),
              np.array(['2000-06-25T12:30:04Z', '2000-06-28T12:50:04Z'],
                       dtype='M8[s]'))
    assert eq(compute(s.truncate(2, 'weeks'), dts),
              np.array(['2000-06-18', '2000-06-18'], dtype='M8[D]'))

    assert into(list, compute(s.truncate(1, 'week'), dts))[0].isoweekday() == 7



def test_hour():
    dts = [datetime(2000, 6, 20,  1, 00, 00),
           datetime(2000, 6, 20, 12, 59, 59),
           datetime(2000, 6, 20, 12, 00, 00),
           datetime(2000, 6, 20, 11, 59, 59)]
    dts = into(np.ndarray, dts)

    assert eq(compute(s.truncate(1, 'hour'), dts),
            into(np.ndarray, [datetime(2000, 6, 20,  1, 0),
                              datetime(2000, 6, 20, 12, 0),
                              datetime(2000, 6, 20, 12, 0),
                              datetime(2000, 6, 20, 11, 0)]))


def test_month():
    dts = [datetime(2000, 7, 1),
           datetime(2000, 6, 30),
           datetime(2000, 6, 1),
           datetime(2000, 5, 31)]
    dts = into(np.ndarray, dts)

    assert eq(compute(s.truncate(1, 'month'), dts),
            into(np.ndarray, [date(2000, 7, 1),
                              date(2000, 6, 1),
                              date(2000, 6, 1),
                              date(2000, 5, 1)]))


def test_truncate_on_np_datetime64_scalar():
    s = symbol('s', 'datetime')
    data = np.datetime64('2000-01-02T12:30:00Z')
    assert compute(s.truncate(1, 'day'), data) == data.astype('M8[D]')


def test_numpy_and_python_datetime_truncate_agree_on_start_of_week():
    s = symbol('s', 'datetime')
    n = np.datetime64('2014-11-11')
    p = datetime(2014, 11, 11)
    expr = s.truncate(1, 'week')
    assert compute(expr, n) == compute(expr, p)
