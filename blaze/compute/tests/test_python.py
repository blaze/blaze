from __future__ import absolute_import, division, print_function

from blaze.compute.python import *
from blaze.expr.table import *
import math

t = TableSymbol('{name: string, amount: int, id: int}')


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]


tbig = TableSymbol('{name: string, sex: string[1], amount: int, id: int}')


databig = [['Alice', 'F', 100, 1],
           ['Alice', 'F', 100, 3],
           ['Drew', 'F', 100, 4],
           ['Drew', 'M', 100, 5],
           ['Drew', 'M', 200, 5]]


def test_table():
    assert compute(t, data) == data


def test_projection():
    assert list(compute(t['name'], data)) == [x[0] for x in data]


def test_eq():
    assert list(compute(t['amount'] == 100, data)) == [x[1] == 100 for x in data]


def test_selection():
    assert list(compute(t[t['amount'] == 0], data)) == \
                [x for x in data if x[1] == 0]
    assert list(compute(t[t['amount'] > 150], data)) == \
                [x for x in data if x[1] > 150]

def test_arithmetic():
    assert list(compute(t['amount'] + t['id'], data)) == \
                [b + c for a, b, c, in data]
    assert list(compute(t['amount'] * t['id'], data)) == \
                [b * c for a, b, c, in data]
    assert list(compute(t['amount'] % t['id'], data)) == \
                [b % c for a, b, c, in data]

def test_unary_ops():
    assert list(compute(exp(t['amount']), data)) == [math.exp(x[1]) for x in data]

def test_reductions():
    assert compute(sum(t['amount']), data) == 100 + 200 + 50
    assert compute(min(t['amount']), data) == 50
    assert compute(max(t['amount']), data) == 200
    assert compute(any(t['amount'] > 150), data) is True
    assert compute(any(t['amount'] > 250), data) is False

def test_mean():
    assert compute(mean(t['amount']), data) == float(100 + 200 + 50) / 3
    assert 50 < compute(std(t['amount']), data) < 100


def test_by():
    assert compute(By(t, t['name'], sum(t['amount'])), data) == \
            {'Alice': 150, 'Bob': 200}


def test_by_big():
    result = compute(By(tbig, tbig[['name', 'sex']], sum(tbig['amount'])),
                     databig)

    expected = {('Alice', 'F'): 200,
                ('Drew', 'F'): 100,
                ('Drew', 'M'): 300}

    assert result == expected
