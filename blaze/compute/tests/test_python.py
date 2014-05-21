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
    assert compute(nunique(t['amount']), data) == 3
    assert compute(nunique(t['name']), data) == 2
    assert compute(count(t['amount']), data) == 3
    assert compute(any(t['amount'] > 150), data) is True
    assert compute(any(t['amount'] > 250), data) is False

def test_mean():
    assert compute(mean(t['amount']), data) == float(100 + 200 + 50) / 3
    assert 50 < compute(std(t['amount']), data) < 100


def test_by_one():
    print(compute(By(t, t['name'], t['amount'].sum()), data))
    assert set(compute(By(t, t['name'], t['amount'].sum()), data)) == \
            set([('Alice', 150), ('Bob', 200)])


def test_by_two():
    result = compute(By(tbig, tbig[['name', 'sex']], tbig['amount'].sum()),
                     databig)

    expected = [(('Alice', 'F'), 200),
                (('Drew', 'F'), 100),
                (('Drew', 'M'), 300)]

    print(result)
    assert set(result) == set(expected)


def test_by_three():
    result = compute(By(tbig,
                        tbig[['name', 'sex']],
                        (tbig['id'] + tbig['amount']).sum()),
                     databig)

    expected = [(('Alice', 'F'), 204),
                (('Drew', 'F'), 104),
                (('Drew', 'M'), 310)]

    print(result)
    assert set(result) == set(expected)


def test_works_on_generators():
    assert list(compute(t['amount'], iter(data))) == \
            [x[1] for x in data]
    assert list(compute(t['amount'], (i for i in data))) == \
            [x[1] for x in data]


def test_join():
    left = [('Alice', 100), ('Bob', 200)]
    right = [('Alice', 1), ('Bob', 2)]

    L = TableSymbol('{name: string, amount: int}')
    R = TableSymbol('{name: string, id: int}')
    joined = Join(L, R, 'name')

    assert dshape(joined.schema) == \
            dshape('{name: string, amount: int, id: int}')

    result = list(compute(joined, {L: left, R: right}))

    expected = [('Alice', 100, 1), ('Bob', 200, 2)]

    assert result == expected


def test_sort():
    assert list(compute(t.sort('amount'), data)) == \
            sorted(data, key=lambda x: x[1], reverse=False)

    assert list(compute(t.sort('amount', ascending=True), data)) == \
            sorted(data, key=lambda x: x[1], reverse=False)

    assert list(compute(t.sort(['amount', 'id']), data)) == \
            sorted(data, key=lambda x: (x[1], x[2]), reverse=False)


def test_head():
    assert list(compute(t.head(1), data)) == [data[0]]
