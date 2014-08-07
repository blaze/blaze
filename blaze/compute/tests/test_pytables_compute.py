from __future__ import absolute_import, division, print_function

import pytest
tables = pytest.importorskip('tables')

import numpy as np
import tempfile
from contextlib import contextmanager
import os

from blaze.compute.core import compute
from blaze.compute.pytables import *
from blaze.compute.numpy import *
from blaze.expr.table import *
from blaze.compatibility import xfail

t = TableSymbol('t', '{id: int, name: string, amount: int}')

x = np.array([(1, 'Alice', 100),
              (2, 'Bob', -200),
              (3, 'Charlie', 300),
              (4, 'Denis', 400),
              (5, 'Edith', -500)],
            dtype=[('id', '<i8'), ('name', 'S7'), ('amount', '<i8')])

@contextmanager
def data():
    filename = tempfile.mktemp()
    f = tables.open_file(filename, 'w')
    d = f.createTable('/', 'title',  x)

    yield d

    d.close()
    f.close()
    os.remove(filename)


def eq(a, b):
    return (a == b).all()


def test_table():
    with data() as d:
        assert compute(t, d) == d


def test_projection():
    with data() as d:
        assert eq(compute(t['name'], d), x['name'])


@xfail(reason="ColumnWise not yet supported")
def test_eq():
    with data() as d:
        assert eq(compute(t['amount'] == 100, d),
                  x['amount'] == 100)


def test_selection():
    with data() as d:
        assert eq(compute(t[t['amount'] == 100], d), x[x['amount'] == 0])
        assert eq(compute(t[t['amount'] < 0], d), x[x['amount'] < 0])


@xfail(reason="ColumnWise not yet supported")
def test_arithmetic():
    with data() as d:
        assert eq(compute(t['amount'] + t['id'], d),
                  x['amount'] + x['id'])
        assert eq(compute(t['amount'] * t['id'], d),
                  x['amount'] * x['id'])
        assert eq(compute(t['amount'] % t['id'], d),
                  x['amount'] % x['id'])

def test_Reductions():
    with data() as d:
        assert compute(t['amount'].count(), d) == len(x['amount'])


@xfail(reason="TODO: sorting could work if on indexed column")
def test_sort():
    with data() as d:
        assert eq(compute(t.sort('amount'), d),
                  np.sort(x, order='amount'))

        assert eq(compute(t.sort('amount', ascending=False), d),
                  np.sort(x, order='amount')[::-1])

        assert eq(compute(t.sort(['amount', 'id']), d),
                  np.sort(x, order=['amount', 'id']))


def test_head():
    with data() as d:
        assert eq(compute(t.head(2), d),
                  x[:2])
