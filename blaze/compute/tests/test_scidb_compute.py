from __future__ import absolute_import, division, print_function

import numpy as np
from scidbpy import connect, SciDBArray

from blaze.compute.core import compute
from blaze.expr import TableSymbol, union, by, exp
from blaze import into

sdb = connect()

t = TableSymbol('t', '{id: int, name: string, amount: int}')

x_np = np.array([(1, 'Alice', 100),
                 (2, 'Bob', -200),
                 (3, 'Charlie', 300),
                 (4, 'Denis', 400),
                 (5, 'Edith', -500)],
                dtype=[('id', 'i8'), ('name', 'S7'), ('amount', 'i8')])
x = sdb.from_array(x_np)


def _coerce_lists(a, b):
    # cast arrays into lists, which ignores difference
    # between string and unicode

    if isinstance(a, SciDBArray):
        a = a.toarray()
    if isinstance(b, SciDBArray):
        b = b.toarray()
    a = np.asarray(a).tolist()
    b = np.asarray(b).tolist()

    return a, b


def eq(a, b):
    a, b = _coerce_lists(a, b)
    np.testing.assert_array_equal(a, b)
    return True


def close(a, b):
    a, b = _coerce_lists(a, b)
    np.testing.assert_allclose(a, b)
    return True


def test_table():
    assert eq(compute(t, x), x)


def test_projection():
    assert eq(compute(t['name'], x), x['name'])


def test_eq():
    assert eq(compute(t['amount'] == 100, x),
              x['amount'] == 100)


def test_selection():
    assert eq(compute(t[t['amount'] == 100], x), x_np[x_np['amount'] == 100])
    assert eq(compute(t[t['amount'] < 0], x), x_np[x_np['amount'] < 0])


def test_arithmetic():
    assert eq(compute(t['amount'] + t['id'], x),
              x['amount'] + x['id'])
    assert eq(compute(t['amount'] * t['id'], x),
              x['amount'] * x['id'])
    assert eq(compute(t['amount'] % t['id'], x),
              x['amount'] % x['id'])


def test_UnaryOp():
    assert eq(compute(exp(t['amount']), x),
              np.exp(x['amount'].toarray()))


def test_Neg():
    assert eq(compute(-t['amount'], x),
              -x_np['amount'])


def test_invert_not():
    assert eq(compute(~(t.amount > 0), x),
              ~(x['amount'] > 0))


def test_union_1d():
    t = TableSymbol('t', '{x: int}', iscolumn=True)
    x = np.array([1, 2, 3])
    assert eq(compute(union(t, t), x), np.array([1, 2, 3, 1, 2, 3]))


def test_union():
    result = compute(union(t, t), x)
    assert result.shape == (x.shape[0] * 2,)
    assert eq(result[:5], x)
    assert eq(result[5:], x)
    result = compute(union(t.id, t.id), x)
    assert eq(result, np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))


def test_Reductions():
    assert eq(compute(t['amount'].mean(), x), x_np['amount'].mean())
    assert eq(compute(t['amount'].count(), x), len(x_np['amount']))
    assert eq(compute(t['amount'].sum(), x), x_np['amount'].sum())
    assert eq(compute(t['amount'].min(), x), x_np['amount'].min())
    assert eq(compute(t['amount'].max(), x), x_np['amount'].max())
    assert eq(compute(t['amount'].nunique(), x), len(np.unique(x_np['amount'])))
    assert close(compute(t['amount'].var(), x), x_np['amount'].var())
    assert close(compute(t['amount'].std(), x), x_np['amount'].std())
    assert close(compute(t['amount'].var(unbiased=True), x), x_np['amount'].var(ddof=1))
    assert close(compute(t['amount'].std(unbiased=True), x), x_np['amount'].std(ddof=1))
    assert eq(compute((t['amount'] > 150).any(), x), True)
    assert eq(compute((t['amount'] > 250).all(), x), False)


def test_Distinct():
    x = np.array([('Alice', 100),
                  ('Alice', -200),
                  ('Bob', 100),
                  ('Bob', 100)],
                 dtype=[('name', 'S5'), ('amount', 'i8')])

    t = TableSymbol('t', '{name: string, amount: int64}')

    assert eq(compute(t['name'].distinct(), x),
              np.unique(x['name']))
    assert eq(compute(t.distinct(), x),
              np.unique(x))


def test_sort():
    assert eq(compute(t.sort('amount'), x),
              np.sort(x_np, order='amount'))

    assert eq(compute(t.sort('amount', ascending=False), x),
              np.sort(x_np, order='amount')[::-1])

    assert eq(compute(t.sort(['amount', 'id']), x),
              np.sort(x_np, order=['amount', 'id']))


def test_head():
    # cast to lists, because SciDB casts strings to unicode and trips array inequality
    assert eq(compute(t.head(2), x),
              x_np[:2])


def test_label():
    expected = x_np['amount'] * 10
    expected = np.array(expected, dtype=[('foo', 'i8')])

    result = compute((t['amount'] * 10).label('foo'), x)

    assert result.att_names == ['foo']
    assert eq(result, x_np['amount'] * 10)


def test_relabel():
    expected = np.array([(1, 'Alice', 100),
                        (2, 'Bob', -200),
                        (3, 'Charlie', 300),
                        (4, 'Denis', 400),
                        (5, 'Edith', -500)],
                        dtype=[('ID', 'i8'), ('NAME', 'S7'), ('amount', 'i8')])
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), x)
    assert result.dtype.names == expected.dtype.names
    assert eq(result, expected)


def test_by():
    expr = by(t.amount > 0, t.id.count())
    result = compute(expr, x)

    assert set(map(tuple, into([], result))) == set([(False, 2), (True, 3)])


def test_by_column():
    expr = by(t.amount, t.id.count())
    result = compute(expr, x)

    assert set(map(tuple, into([], result))) == set([(-500, 1), (-200, 1), (100, 1), (300, 1), (400, 1)])
