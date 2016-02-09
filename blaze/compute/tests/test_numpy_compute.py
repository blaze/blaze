from __future__ import absolute_import, division, print_function

import pytest

import itertools

import numpy as np
import pandas as pd

from datetime import datetime, date

from blaze.compute.core import compute, compute_up
from blaze.expr import symbol, by, exp, summary, Broadcast, join, concat
from blaze.expr import greatest, least, coalesce
from blaze import sin
import blaze
from odo import into
from datashape import discover, to_numpy, dshape


x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
             dtype=[('id', 'i8'), ('name', 'S7'), ('amount', 'i8')])

t = symbol('t', discover(x))


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

    assert eq(compute(abs(-t['amount']), x),
              abs(-x['amount']))


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
    assert compute(t['amount'][0], x) == x['amount'][0]
    assert compute(t['amount'][-1], x) == x['amount'][-1]


def test_count_string():
    s = symbol('name', 'var * ?string')
    x = np.array(['Alice', np.nan, 'Bob', 'Denis', 'Edith'], dtype='object')
    assert compute(s.count(), x) == 4


def test_reductions_on_recarray():
    assert compute(t.count(), x) == len(x)


def test_count_nan():
    t = symbol('t', '3 * ?real')
    x = np.array([1.0, np.nan, 2.0])
    assert compute(t.count(), x) == 2


def test_distinct():
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


def test_distinct_on_recarray():
    rec = pd.DataFrame(
        [[0, 1],
         [0, 2],
         [1, 1],
         [1, 2]],
        columns=('a', 'b'),
    ).to_records(index=False)

    s = symbol('s', discover(rec))
    assert (
        compute(s.distinct('a'), rec) ==
        pd.DataFrame(
            [[0, 1],
             [1, 1]],
            columns=('a', 'b'),
        ).to_records(index=False)
    ).all()


def test_distinct_on_structured_array():
    arr = np.array(
        [(0., 1.),
         (0., 2.),
         (1., 1.),
         (1., 2.)],
        dtype=[('a', 'f4'), ('b', 'f4')],
    )

    s = symbol('s', discover(arr))
    assert(
        compute(s.distinct('a'), arr) ==
        np.array([(0., 1.), (1., 1.)], dtype=arr.dtype)
    ).all()


def test_distinct_on_str():
    rec = pd.DataFrame(
        [['a', 'a'],
         ['a', 'b'],
         ['b', 'a'],
         ['b', 'b']],
        columns=('a', 'b'),
    ).to_records(index=False).astype([('a', '<U1'), ('b', '<U1')])

    s = symbol('s', discover(rec))
    assert (
        compute(s.distinct('a'), rec) ==
        pd.DataFrame(
            [['a', 'a'],
             ['b', 'a']],
            columns=('a', 'b'),
        ).to_records(index=False).astype([('a', '<U1'), ('b', '<U1')])
    ).all()


def test_sort():
    assert eq(compute(t.sort('amount'), x),
              np.sort(x, order='amount'))

    assert eq(compute(t.sort('amount', ascending=False), x),
              np.sort(x, order='amount')[::-1])

    assert eq(compute(t.sort(['amount', 'id']), x),
              np.sort(x, order=['amount', 'id']))

    assert eq(compute(t.amount.sort(), x),
              np.sort(x['amount']))


def test_head():
    assert eq(compute(t.head(2), x),
              x[:2])


def test_tail():
    assert eq(compute(t.tail(2), x),
              x[-2:])


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


ax = np.arange(30, dtype='f4').reshape((5, 3, 2))

a = symbol('a', discover(ax))

def test_slice():
    inds = [0, slice(2), slice(1, 3), slice(None, None, 2), [1, 2, 3],
            (0, 1), (0, slice(1, 3)), (slice(0, 3), slice(3, 1, -1)),
            (0, [1, 2])]
    for s in inds:
        assert (compute(a[s], ax) == ax[s]).all()


def test_array_reductions():
    for axis in [None, 0, 1, (0, 1), (2, 1)]:
        assert eq(compute(a.sum(axis=axis), ax), ax.sum(axis=axis))
        assert eq(compute(a.std(axis=axis), ax), ax.std(axis=axis))


def test_array_reductions_with_keepdims():
    for axis in [None, 0, 1, (0, 1), (2, 1)]:
        assert eq(compute(a.sum(axis=axis, keepdims=True), ax),
                 ax.sum(axis=axis, keepdims=True))


def test_summary_on_ndarray():
    assert compute(summary(total=a.sum(), min=a.min()), ax) == \
            (ax.min(), ax.sum())

    result = compute(summary(total=a.sum(), min=a.min(), keepdims=True), ax)
    expected = np.array([(ax.min(), ax.sum())],
                        dtype=[('min', 'float32'), ('total', 'float64')])
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


def test_add_multiple_ndarrays():
    a = symbol('a', '5 * 4 * int64')
    b = symbol('b', '5 * 4 * float32')
    x = np.arange(9, dtype='int64').reshape(3, 3)
    y = (x + 1).astype('float32')
    expr = sin(a) + 2 * b
    scope = {a: x, b: y}
    expected = sin(x) + 2 * y

    # check that we cast correctly
    assert expr.dshape == dshape('5 * 4 * float64')

    np.testing.assert_array_equal(compute(expr, scope), expected)
    np.testing.assert_array_equal(compute(expr, scope, optimize=False),
                                  expected)


nA = np.arange(30, dtype='f4').reshape((5, 6))
ny = np.arange(6, dtype='f4')

A = symbol('A', discover(nA))
y = symbol('y', discover(ny))


def test_transpose():
    assert eq(compute(A.T, nA), nA.T)
    assert eq(compute(A.transpose((0, 1)), nA), nA)


def test_dot():
    assert eq(compute(y.dot(y), {y: ny}), np.dot(ny, ny))
    assert eq(compute(A.dot(y), {A: nA, y: ny}), np.dot(nA, ny))


def test_subexpr_datetime():
    data = pd.date_range(start='01/01/2010', end='01/04/2010', freq='D').values
    s = symbol('s', discover(data))
    result = compute(s.truncate(days=2).day, data)
    expected = np.array([31, 2, 2, 4])
    np.testing.assert_array_equal(result, expected)


def test_mixed_types():
    x = np.array([[(4, 180), (4, 184), (4, 188), (4, 192), (4, 196)],
                  [(4, 660), (4, 664), (4, 668), (4, 672), (4, 676)],
                  [(4, 1140), (4, 1144), (4, 1148), (4, 1152), (4, 1156)],
                  [(4, 1620), (4, 1624), (4, 1628), (4, 1632), (4, 1636)],
                  [(4, 2100), (4, 2104), (4, 2108), (4, 2112), (4, 2116)]],
                 dtype=[('count', '<i4'), ('total', '<i8')])
    aggregate = symbol('aggregate', discover(x))
    result = compute(aggregate.total.sum(axis=(0,)) /
                     aggregate['count'].sum(axis=(0,)), x)
    expected = (x['total'].sum(axis=0, keepdims=True) /
                x['count'].sum(axis=0, keepdims=True)).squeeze()
    np.testing.assert_array_equal(result, expected)


def test_broadcast_compute_against_numbers_and_arrays():
    A = symbol('A', '5 * float32')
    a = symbol('a', 'float32')
    b = symbol('b', 'float32')
    x = np.arange(5, dtype='f4')
    expr = Broadcast((A, b), (a, b), a + b)
    result = compute(expr, {A: x, b: 10})
    assert eq(result, x + 10)


def test_map():
    pytest.importorskip('numba')
    a = np.arange(10.0)
    f = lambda x: np.sin(x) + 1.03 * np.cos(x) ** 2
    x = symbol('x', discover(a))
    expr = x.map(f, 'float64')
    result = compute(expr, a)
    expected = f(a)

    # make sure we're not going to pandas here
    assert type(result) == np.ndarray
    assert type(result) == type(expected)

    np.testing.assert_array_equal(result, expected)


def test_vector_norm():
    x = np.arange(30).reshape((5, 6))
    s = symbol('x', discover(x))

    assert eq(compute(s.vnorm(), x),
              np.linalg.norm(x))
    assert eq(compute(s.vnorm(ord=1), x),
              np.linalg.norm(x.flatten(), ord=1))
    assert eq(compute(s.vnorm(ord=4, axis=0), x),
              np.linalg.norm(x, ord=4, axis=0))

    expr = s.vnorm(ord=4, axis=0, keepdims=True)
    assert expr.shape == compute(expr, x).shape


def test_join():
    cities = np.array([('Alice', 'NYC'),
                       ('Alice', 'LA'),
                       ('Bob', 'Chicago')],
                      dtype=[('name', 'S7'), ('city', 'O')])

    c = symbol('cities', discover(cities))

    expr = join(t, c, 'name')
    result = compute(expr, {t: x, c: cities})
    assert (b'Alice', 1, 100, 'LA') in into(list, result)


def test_query_with_strings():
    b = np.array([('a', 1), ('b', 2), ('c', 3)],
                 dtype=[('x', 'S1'), ('y', 'i4')])

    s = symbol('s', discover(b))
    assert compute(s[s.x == b'b'], b).tolist() == [(b'b', 2)]


@pytest.mark.parametrize('keys', [['a'], list('bc')])
def test_isin(keys):
    b = np.array([('a', 1), ('b', 2), ('c', 3), ('a', 4), ('c', 5), ('b', 6)],
                 dtype=[('x', 'S1'), ('y', 'i4')])

    s = symbol('s', discover(b))
    result = compute(s.x.isin(keys), b)
    expected = np.in1d(b['x'], keys)
    np.testing.assert_array_equal(result, expected)


def test_nunique_recarray():
    b = np.array([('a', 1), ('b', 2), ('c', 3), ('a', 4), ('c', 5), ('b', 6),
                  ('a', 1), ('b', 2)],
                 dtype=[('x', 'S1'), ('y', 'i4')])
    s = symbol('s', discover(b))
    expr = s.nunique()
    assert compute(expr, b) == len(np.unique(b))


def test_str_repeat():
    a = np.array(('a', 'b', 'c'))
    s = symbol('s', discover(a))
    expr = s.repeat(3)
    assert all(compute(expr, a) == np.char.multiply(a, 3))


def test_str_interp():
    a = np.array(('%s', '%s', '%s'))
    s = symbol('s', discover(a))
    expr = s.interp(1)
    assert all(compute(expr, a) == np.char.mod(a, 1))


def test_timedelta_arith():
    dates = np.arange('2014-01-01', '2014-02-01', dtype='datetime64')
    delta = np.timedelta64(1, 'D')
    sym = symbol('s', discover(dates))
    assert (compute(sym + delta, dates) == dates + delta).all()
    assert (compute(sym - delta, dates) == dates - delta).all()
    assert (
        compute(sym - (sym - delta), dates) ==
        dates - (dates - delta)
    ).all()


def test_coerce():
    x = np.arange(1, 3)
    s = symbol('s', discover(x))
    np.testing.assert_array_equal(compute(s.coerce('float64'), x),
                                  np.arange(1.0, 3.0))


def test_concat_arr():
    s_data = np.arange(15)
    t_data = np.arange(15, 30)

    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))

    assert (
        compute(concat(s, t), {s: s_data, t: t_data}) ==
        np.arange(30)
    ).all()


def test_concat_mat():
    s_data = np.arange(15).reshape(5, 3)
    t_data = np.arange(15, 30).reshape(5, 3)

    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))

    assert (
        compute(concat(s, t), {s: s_data, t: t_data}) ==
        np.arange(30).reshape(10, 3)
    ).all()
    assert (
        compute(concat(s, t, axis=1), {s: s_data, t: t_data}) ==
        np.concatenate((s_data, t_data), axis=1)
    ).all()


@pytest.mark.parametrize('dtype', ['int64', 'float64'])
def test_least(dtype):
    s_data = np.arange(15, dtype=dtype).reshape(5, 3)
    t_data = np.arange(15, 30, dtype=dtype).reshape(5, 3)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    expr = least(s, t)
    result = compute(expr, {s: s_data, t: t_data})
    expected = np.minimum(s_data, t_data)
    assert np.all(result == expected)


@pytest.mark.parametrize('dtype', ['int64', 'float64'])
def test_least_mixed(dtype):
    s_data = np.array([2, 1], dtype=dtype)
    t_data = np.array([1, 2], dtype=dtype)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    expr = least(s, t)
    result = compute(expr, {s: s_data, t: t_data})
    expected = np.minimum(s_data, t_data)
    assert np.all(result == expected)


@pytest.mark.parametrize('dtype', ['int64', 'float64'])
def test_greatest(dtype):
    s_data = np.arange(15, dtype=dtype).reshape(5, 3)
    t_data = np.arange(15, 30, dtype=dtype).reshape(5, 3)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    expr = greatest(s, t)
    result = compute(expr, {s: s_data, t: t_data})
    expected = np.maximum(s_data, t_data)
    assert np.all(result == expected)


@pytest.mark.parametrize('dtype', ['int64', 'float64'])
def test_greatest_mixed(dtype):
    s_data = np.array([2, 1], dtype=dtype)
    t_data = np.array([1, 2], dtype=dtype)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    expr = greatest(s, t)
    result = compute(expr, {s: s_data, t: t_data})
    expected = np.maximum(s_data, t_data)
    assert np.all(result == expected)


binary_name_map = {
    'atan2': 'arctan2'
}


@pytest.mark.parametrize(
    ['func', 'kwargs'],
    itertools.product(['copysign', 'ldexp'], [dict(optimize=False), dict()])
    )
def test_binary_math(func, kwargs):
    s_data = np.arange(15).reshape(5, 3)
    t_data = np.arange(15, 30).reshape(5, 3)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    scope = {s: s_data, t: t_data}
    result = compute(getattr(blaze, func)(s, t), scope, **kwargs)
    expected = getattr(np, binary_name_map.get(func, func))(s_data, t_data)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ['func', 'kwargs'],
    itertools.product(['atan2', 'hypot'], [dict(optimize=False), dict()])
)
def test_floating_binary_math(func, kwargs):
    s_data = np.arange(15).reshape(5, 3)
    t_data = np.arange(15, 30).reshape(5, 3)
    s = symbol('s', discover(s_data))
    t = symbol('t', discover(t_data))
    scope = {s: s_data, t: t_data}
    result = compute(getattr(blaze, func)(s, t), scope, **kwargs)
    expected = getattr(np, binary_name_map.get(func, func))(s_data, t_data)
    np.testing.assert_allclose(result, expected)


def test_selection_inner_inputs():
    s_data = np.arange(5).reshape(5, 1)
    t_data = np.arange(5).reshape(5, 1)

    s = symbol('s', 'var * {a: int64}')
    t = symbol('t', 'var * {a: int64}')

    assert (
        compute(s[s.a == t.a], {s: s_data, t: t_data}) ==
        s_data
    ).all()


def test_coalesce():
    data = np.array([0, None, 1, None, 2, None])

    s = symbol('s', 'var * ?int')
    t = symbol('t', 'int')
    u = symbol('u', '?int')
    v = symbol('v', 'var * int')
    w = symbol('w', 'var * ?int')

    # array to scalar
    np.testing.assert_array_equal(
        compute(coalesce(s, t), {s: data, t: -1}),
        np.array([0, -1, 1, -1, 2, -1]),
    )
    # array to scalar with NULL
    np.testing.assert_array_equal(
        compute(coalesce(s, u), {s: data, u: None}),
        np.array([0, None, 1, None, 2, None], dtype=object)
    )
    # array to array
    np.testing.assert_array_equal(
        compute(coalesce(s, v), {
            s: data, v: np.array([-1, -2, -3, -4, -5, -6]),
        }),
        np.array([0, -2, 1, -4, 2, -6])
    )
    # array to array with NULL
    np.testing.assert_array_equal(
        compute(coalesce(s, w), {
            s: data, w: np.array([-1, None, -3, -4, -5, -6]),
        }),
        np.array([0, None, 1, -4, 2, -6]),
    )
