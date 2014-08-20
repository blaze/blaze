from __future__ import absolute_import, division, print_function

import pytest
tables = pytest.importorskip('tables')

import numpy as np
import tempfile

from blaze.compute.core import compute
from blaze.expr import TableSymbol


t = TableSymbol('t', '{id: int, name: string, amount: int}')

x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
             dtype=[('id', '<i8'), ('name', 'S7'), ('amount', '<i8')])


@pytest.yield_fixture
def data():
    with tempfile.NamedTemporaryFile() as fobj:
        f = tables.open_file(fobj.name, mode='w')
        d = f.create_table('/', 'title',  x)
        yield d
        d.close()
        f.close()


def eq(a, b):
    return (a == b).all()


def test_table(data):
    assert compute(t, data) == data


def test_single_column(data):
    assert eq(compute(t['name'], data), x['name'])


def test_projection(data):
    assert eq(compute(t[['name', 'amount']], data), x[['name', 'amount']])


def test_eq(data):
    assert eq(compute(t['amount'] == 100, data), x['amount'] == 100)


def test_scalar_ops(data):
    from operator import add, sub, mul, truediv

    for op in (add, sub, mul, truediv):
        assert eq(compute(op(t.amount, 10), data), op(x['amount'], 10))
        assert eq(compute(op(t.amount, t.id), data), op(x['amount'], x['id']))
        assert eq(compute(op(10.0, t.amount), data), op(10.0, x['amount']))
        assert eq(compute(op(10, t.amount), data), op(10, x['amount']))


def test_failing_floordiv(data):
    from operator import floordiv as op

    with pytest.raises(NotImplementedError):
        assert eq(compute(op(t.amount, 10), data), op(x['amount'], 10))

    with pytest.raises(NotImplementedError):
        assert eq(compute(op(t.amount, t.id), data), op(x['amount'], x['id']))

    with pytest.raises(NotImplementedError):
        assert eq(compute(op(10.0, t.amount), data), op(10.0, x['amount']))

    with pytest.raises(NotImplementedError):
        assert eq(compute(op(10, t.amount), data), op(10, x['amount']))


def test_selection(data):
    assert eq(compute(t[t['amount'] == 100], data), x[x['amount'] == 0])
    assert eq(compute(t[t['amount'] < 0], data), x[x['amount'] < 0])


def test_arithmetic(data):
    assert eq(compute(t['amount'] + t['id'], data), x['amount'] + x['id'])
    assert eq(compute(t['amount'] * t['id'], data), x['amount'] * x['id'])
    assert eq(compute(t['amount'] % t['id'], data), x['amount'] % x['id'])
    assert eq(compute(t['amount'] + t['id'] + 3, data),
              x['amount'] + x['id'] + 3)


class TestReductions(object):
    def test_count(self, data):
        assert compute(t['amount'].count(), data) == len(x['amount'])

    def test_sum(self, data):
        assert compute(t['amount'].sum(), data) == x['amount'].sum()

    def test_mean(self, data):
        assert compute(t['amount'].mean(), data) == x['amount'].mean()

    def test_count_func(self, data):
        from blaze import count
        assert compute(count(t['amount']), data) == len(x['amount'])

    def test_sum_func(self, data):
        from blaze import sum
        assert compute(sum(t['amount']), data) == x['amount'].sum()

    def test_mean_func(self, data):
        from blaze import mean
        assert compute(mean(t['amount']), data) == x['amount'].mean()


class TestSort(object):
    def test_basic(self, data):
        assert eq(compute(t.sort('amount'), data), np.sort(x, order='amount'))

    def test_ascending(self, data):
        assert eq(compute(t.sort('amount', ascending=False), data),
                np.sort(x, order='amount')[::-1])

    def test_multiple_columns(self, data):
        assert eq(compute(t.sort(['amount', 'id']), data),
                  np.sort(x, order=['amount', 'id']))


def test_head(data):
    assert eq(compute(t.head(2), data), x[:2])
