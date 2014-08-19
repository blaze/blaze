from __future__ import absolute_import, division, print_function

import pytest
tables = pytest.importorskip('tables')

import numpy as np
import tempfile
import os

from blaze.compute.core import compute
from blaze.expr import TableSymbol
from blaze.compatibility import xfail

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


def test_projection(data):
    assert eq(compute(t['name'], data), x['name'])


def test_eq(data):
    assert eq(compute(t['amount'] == 100, data), x['amount'] == 100)


def test_selection(data):
    assert eq(compute(t[t['amount'] == 100], data), x[x['amount'] == 0])
    assert eq(compute(t[t['amount'] < 0], data), x[x['amount'] < 0])


def test_arithmetic(data):
    assert eq(compute(t['amount'] + t['id'], data), x['amount'] + x['id'])
    assert eq(compute(t['amount'] * t['id'], data), x['amount'] * x['id'])
    assert eq(compute(t['amount'] % t['id'], data), x['amount'] % x['id'])


def test_reductions(data):
    assert compute(t['amount'].count(), data) == len(x['amount'])


@xfail(reason="TODO: sorting could work if on indexed column")
def test_sort(data):
    assert eq(compute(t.sort('amount'), data), np.sort(x, order='amount'))
    assert eq(compute(t.sort('amount', ascending=False), data),
              np.sort(x, order='amount')[::-1])
    assert eq(compute(t.sort(['amount', 'id']), data),
              np.sort(x, order=['amount', 'id']))


def test_head(data):
    assert eq(compute(t.head(2), data), x[:2])
