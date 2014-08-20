from __future__ import absolute_import, division, print_function

import pytest
tables = pytest.importorskip('tables')

from blaze.compatibility import xfail

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


@pytest.yield_fixture
def csi_data():
    with tempfile.NamedTemporaryFile() as fobj:
        f = tables.open_file(fobj.name, mode='w')
        d = f.create_table('/', 'title', x)
        d.cols.amount.create_csindex()
        d.cols.id.create_csindex()
        yield d
        d.close()
        f.close()


@pytest.yield_fixture
def idx_data():
    with tempfile.NamedTemporaryFile() as fobj:
        f = tables.open_file(fobj.name, mode='w')
        d = f.create_table('/', 'title', x)
        d.cols.amount.create_index()
        d.cols.id.create_index()
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


def test_neg(data):
    assert eq(compute(-t.amount, data), -x['amount'])


def test_failing_floordiv(data):
    from operator import floordiv as op

    with pytest.raises(TypeError):
        assert eq(compute(op(t.amount, 10), data), op(x['amount'], 10))

    with pytest.raises(TypeError):
        assert eq(compute(op(t.amount, t.id), data), op(x['amount'], x['id']))

    with pytest.raises(TypeError):
        assert eq(compute(op(10.0, t.amount), data), op(10.0, x['amount']))

    with pytest.raises(TypeError):
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


class TestTopLevelReductions(object):
    def test_count(self, data):
        from blaze import count
        assert compute(count(t['amount']), data) == len(x['amount'])

    def test_sum(self, data):
        from blaze import sum
        assert compute(sum(t['amount']), data) == x['amount'].sum()

    def test_mean(self, data):
        from blaze import mean
        assert compute(mean(t['amount']), data) == x['amount'].mean()


class TestFailingSort(object):
    """These fail because we haven't created a completely sorted index"""

    def test_basic(self, data):
        with pytest.raises(ValueError):
            compute(t.sort('id'), data)

    @xfail(reason='PyTables does not support multiple column sorting')
    def test_multiple_columns(self, data):
        compute(t.sort(['amount', 'id']), data)

    @xfail(reason='PyTables does not support multiple column sorting')
    def test_multiple_columns_sorted_data(self, csi_data):
        compute(t.sort(['amount', 'id']), csi_data)


class TestCSISort(object):
    def test_basic(self, csi_data):
        assert eq(compute(t.sort('amount'), csi_data),
                  np.sort(x, order='amount'))
        assert eq(compute(t.sort('id'), csi_data),
                  np.sort(x, order='id'))

    def test_ascending(self, csi_data):
        assert eq(compute(t.sort('amount', ascending=False), csi_data),
                  np.sort(x, order='amount')[::-1])
        assert eq(compute(t.sort('amount', ascending=False), csi_data),
                  np.sort(x, order='amount')[::-1])


class TestIndexSort(object):
    """Fails with a partially sorted index"""

    @xfail(reason='PyTables cannot sort with a standard index')
    def test_basic(self, idx_data):
        compute(t.sort('amount'), idx_data)

    @xfail(reason='PyTables cannot sort with a standard index')
    def test_ascending(self, idx_data):
        compute(t.sort('amount', ascending=False), idx_data)


def test_head(data):
    assert eq(compute(t.head(2), data), x[:2])
