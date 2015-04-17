from __future__ import absolute_import, division, print_function

import os
import pytest
import pandas as pd
tb = pytest.importorskip('tables')


try:
    f = pd.HDFStore('foo')
except (RuntimeError, ImportError) as e:
    pytest.skip('skipping test_hdfstore.py %s' % e)
else:
    f.close()
    os.remove('foo')


from blaze.compatibility import xfail

import numpy as np

from blaze.compute.core import compute
from blaze.expr import symbol
from blaze import drop, discover, create_index
from blaze.utils import tmpfile


t = symbol('t', 'var * {id: int, name: string, amount: int}')

x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
             dtype=[('id', '<i8'), ('name', 'S7'), ('amount', '<i8')])


@pytest.yield_fixture
def data():
    with tmpfile('.h5') as filename:
        f = tb.open_file(filename, mode='w')
        d = f.create_table('/', 'title',  x)
        yield d
        d.close()
        f.close()


@pytest.yield_fixture
def csi_data():
    with tmpfile('.h5') as filename:
        f = tb.open_file(filename, mode='w')
        d = f.create_table('/', 'title', x)
        d.cols.amount.create_csindex()
        d.cols.id.create_csindex()
        yield d
        d.close()
        f.close()


@pytest.yield_fixture
def idx_data():
    with tmpfile('.h5') as fn:
        f = tb.open_file(fn, mode='w')
        d = f.create_table('/', 'title', x)
        d.cols.amount.create_index()
        d.cols.id.create_index()
        yield d
        d.close()
        f.close()


def eq(a, b):
    return (a == b).all()


def test_discover_datashape(data):
    ds = discover(data)
    t = symbol('t', ds)
    columns = t.fields
    assert columns is not None


def test_symbol(data):
    assert compute(t, data) == data
    assert isinstance(data, tb.Table)


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


def test_reductions(data):
    assert compute(t['amount'].count(), data) == len(x['amount'])
    assert compute(t['amount'].sum(), data) == x['amount'].sum()
    assert compute(t['amount'].mean(), data) == x['amount'].mean()
    assert compute(t.amount[0], data) == x['amount'][0]
    assert compute(t.amount[-1], data) == x['amount'][-1]


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

    def test_column_expr(self, csi_data):
        assert eq(compute(t.sort(t.amount), csi_data),
                  np.sort(x, order='amount'))
        assert eq(compute(t.sort(t.id), csi_data),
                  np.sort(x, order='id'))

    def test_non_existent_column(self, csi_data):
        with pytest.raises(AssertionError):
            compute(t.sort('not here'), csi_data)

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
    assert eq(compute(t.amount.head(2), data), x['amount'][:2])


@pytest.yield_fixture
def pyt():
    tb = pytest.importorskip('tables')
    fn = 'test.pyt.h5'
    f = tb.open_file(fn, mode='w')
    d = f.create_table('/', 'test', x)
    yield d
    d.close()
    f.close()
    try:
        os.remove(fn)
    except OSError:
        pass


def test_drop(pyt):
    drop(pyt)
    with pytest.raises(tb.ClosedNodeError):
        drop(pyt)


def test_create_index(pyt):
    create_index(pyt, 'id')
    assert 'id' in pyt.colindexes


def test_create_multiple_indexes(pyt):
    create_index(pyt, ['id', 'amount'])
    assert len(pyt.colindexes) == 2
    assert 'id' in pyt.colindexes
    assert 'amount' in pyt.colindexes


def test_create_multiple_indexes_fails(pyt):
    with pytest.raises(ValueError):
        create_index(pyt, ['id', 'blarg'])

    with pytest.raises(ValueError):
        create_index(pyt, ['foo', 'bar'])


def test_create_index_fails(pyt):
    with pytest.raises(AttributeError):
        create_index(pyt, 'no column here!')


def test_nrows():
    assert compute(t.nrows, x) == len(x)


def test_nelements():
    assert compute(t.nelements(axis=0), x) == len(x)
    assert compute(t.nelements(), x) == len(x)
