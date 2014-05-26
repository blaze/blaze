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


def test_graph_double_join():
    idx = [['A', 1],
           ['B', 2],
           ['C', 3],
           ['D', 4],
           ['E', 5],
           ['F', 6]]

    arc = [[1, 3],
           [2, 3],
           [4, 3],
           [5, 3],
           [3, 1],
           [2, 1],
           [5, 1],
           [1, 6],
           [2, 6],
           [4, 6]]

    wanted = [['A'],
              ['F']]

    t_idx = TableSymbol('{name: string, b: int32}')
    t_arc = TableSymbol('{a: int32, b: int32}')
    t_wanted = TableSymbol('{name: string}')

    j = Join(Join(t_idx, t_arc, 'b'), t_wanted, 'name')[['name', 'a', 'b']]

    result = compute(j, {t_idx: idx, t_arc: arc, t_wanted: wanted})
    result = set(map(tuple, result))
    expected = set([('A', 3, 1),
                    ('A', 2, 1),
                    ('A', 5, 1),
                    ('F', 1, 6),
                    ('F', 2, 6),
                    ('F', 4, 6)])

    assert result == expected


def test_label():
    assert list(compute((t['amount'] * 1).label('foo'), data)) == \
            list(compute((t['amount'] * 1), data))


def test_relabel_join():
    names = TableSymbol('{first: string, last: string}')

    siblings = Join(names.relabel({'first': 'left'}),
                    names.relabel({'first': 'right'}),
                    'last')[['left', 'right']]

    data = [('Alice', 'Smith'),
            ('Bob', 'Jones'),
            ('Charlie', 'Smith')]

    print(set(compute(siblings, {names: data})))
    assert ('Alice', 'Charlie') in set(compute(siblings, {names: data}))
    assert ('Alice', 'Bob') not in set(compute(siblings, {names: data}))


def test_map_column():
    inc = lambda x: x + 1
    result = compute(t['amount'].map(inc), data) == [x[1] + 1 for x in data]


def test_map_column():
    inc = lambda x: x + 1
    result = compute(t.map(lambda _, amt, id: amt + id), data) == \
            [x[1] + x[2] for x in data]
