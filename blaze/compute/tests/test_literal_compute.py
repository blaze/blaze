import numpy as np
import pytest

from datashape import dshape
from odo import into

from blaze.compute import compute
from blaze.expr import data, literal, symbol


tdata = (('Alice', 100),
         ('Bob', 200))

L = [[1, 'Alice',   100],
     [2, 'Bob',    -200],
     [3, 'Charlie', 300],
     [4, 'Denis',   400],
     [5, 'Edith',  -500]]

l = literal(tdata)
nl = literal(tdata, name='amounts')
t = data(tdata, fields=['name', 'amount'])


def test_resources_fail():
    t = symbol('t', 'var * {x: int, y: int}')
    d = t[t['x'] > 100]
    with pytest.raises(ValueError):
        compute(d)


def test_compute_on_Data_gives_back_data():
    assert compute(data([1, 2, 3])) == [1, 2, 3]


def test_compute_on_literal_gives_back_data():
    assert compute(literal([1, 2, 3])) == [1, 2, 3]


def test_compute():
    assert list(compute(t['amount'] + 1)) == [101, 201]


def test_create_with_schema():
    t = data(tdata, schema='{name: string, amount: float32}')
    assert t.schema == dshape('{name: string, amount: float32}')


def test_create_with_raw_data():
    t = data(tdata, fields=['name', 'amount'])
    assert t.schema == dshape('{name: string, amount: int64}')
    assert t.name
    assert t.data == tdata


def test_iter():
    x = np.ones(4)
    d = data(x)
    assert list(d + 1) == [2, 2, 2, 2]


def test_into():
    assert into(list, t) == into(list, tdata)
