from __future__ import absolute_import, division, print_function
import math
import itertools
import operator

import blaze
from blaze.compute.python import nunique, mean, std, rrowfunc
from blaze import dshape
from blaze.compute.core import compute, compute_one
from blaze.expr import (TableSymbol, by, union, merge, join, count, Distinct,
                        Apply, sum, min, max, any, exp, summary, ScalarSymbol,
                        count, scalar)

from blaze import cos, sin
from blaze.compatibility import builtins
from blaze.utils import raises

t = TableSymbol('t', '{name: string, amount: int, id: int}')


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]


tbig = TableSymbol('tbig', '{name: string, sex: string[1], amount: int, id: int}')


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
    for op in ('cos', 'sin', 'exp', 'ceil', 'floor', 'trunc', 'isnan'):
        f = getattr(blaze, op)
        pyf = getattr(math, op)
        result = list(compute(f(t['amount']), data))
        assert result == [pyf(x[1]) for x in data]


def test_neg():
    assert list(compute(-t['amount'], data)) == [-x[1] for x in data]


def test_reductions():
    assert compute(sum(t['amount']), data) == 100 + 200 + 50
    assert compute(min(t['amount']), data) == 50
    assert compute(max(t['amount']), data) == 200
    assert compute(nunique(t['amount']), data) == 3
    assert compute(nunique(t['name']), data) == 2
    assert compute(count(t['amount']), data) == 3
    assert compute(any(t['amount'] > 150), data) is True
    assert compute(any(t['amount'] > 250), data) is False


def reduction_runner(funcs):
    from blaze.compatibility import builtins as bts
    exprs = sum, min, max
    for blaze_expr, py_func in itertools.product(exprs, funcs):
        f = getattr(operator, py_func)
        reduc_f = getattr(bts, blaze_expr.__name__)
        ground_truth = f(reduc_f([100, 200, 50]), 5)
        assert compute(f(blaze_expr(t['amount']), 5), data) == ground_truth


def test_reduction_arithmetic():
    funcs = 'add', 'mul'
    reduction_runner(funcs)


def test_reduction_compare():
    funcs = 'eq', 'ne', 'lt', 'gt', 'le', 'ge'
    reduction_runner(funcs)


def test_mean():
    assert compute(mean(t['amount']), data) == float(100 + 200 + 50) / 3
    assert 50 < compute(std(t['amount']), data) < 100


def test_by_no_grouper():
    names = t['name']
    assert set(compute(by(names, names, names.count()), data)) == \
            set([('Alice', 2), ('Bob', 1)])


def test_by_one():
    print(compute(by(t, t['name'], t['amount'].sum()), data))
    assert set(compute(by(t, t['name'], t['amount'].sum()), data)) == \
            set([('Alice', 150), ('Bob', 200)])


def test_by_compound_apply():
    print(compute(by(t, t['name'], (t['amount'] + 1).sum()), data))
    assert set(compute(by(t, t['name'], (t['amount'] + 1).sum()), data)) == \
            set([('Alice', 152), ('Bob', 201)])


def test_by_two():
    result = compute(by(tbig, tbig[['name', 'sex']], tbig['amount'].sum()),
                     databig)

    expected = [('Alice', 'F', 200),
                ('Drew', 'F', 100),
                ('Drew', 'M', 300)]

    print(set(result))
    assert set(result) == set(expected)


def test_by_three():
    result = compute(by(tbig,
                        tbig[['name', 'sex']],
                        (tbig['id'] + tbig['amount']).sum()),
                     databig)

    expected = [('Alice', 'F', 204),
                ('Drew', 'F', 104),
                ('Drew', 'M', 310)]

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

    L = TableSymbol('L', '{name: string, amount: int}')
    R = TableSymbol('R', '{name: string, id: int}')
    joined = join(L, R, 'name')

    assert dshape(joined.schema) == \
            dshape('{name: string, amount: int, id: int}')

    result = list(compute(joined, {L: left, R: right}))

    expected = [('Alice', 100, 1), ('Bob', 200, 2)]

    assert result == expected


def test_outer_join():
    left = [(1, 'Alice', 100),
            (2, 'Bob', 200),
            (4, 'Dennis', 400)]
    right = [('NYC', 1),
             ('Boston', 1),
             ('LA', 3),
             ('Moscow', 4)]

    L = TableSymbol('L', '{id: int, name: string, amount: real}')
    R = TableSymbol('R', '{city: string, id: int}')

    assert set(compute(join(L, R), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (4, 'Dennis', 400, 'Moscow')])

    assert set(compute(join(L, R, how='left'), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (4, 'Dennis', 400, 'Moscow')])

    assert set(compute(join(L, R, how='right'), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (3, None, None, 'LA'),
             (4, 'Dennis', 400, 'Moscow')])

    assert set(compute(join(L, R, how='outer'), {L: left, R: right})) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (3, None, None, 'LA'),
             (4, 'Dennis', 400, 'Moscow')])


def test_multi_column_join():
    left = [(1, 2, 3),
            (2, 3, 4),
            (1, 3, 5)]
    right = [(1, 2, 30),
             (1, 3, 50),
             (1, 3, 150)]

    L = TableSymbol('L', '{x: int, y: int, z: int}')
    R = TableSymbol('R', '{x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    print(list(compute(j, {L: left, R: right})))
    assert list(compute(j, {L: left, R: right})) == [(1, 2, 3, 30),
                                                     (1, 3, 5, 50),
                                                     (1, 3, 5, 150)]


def test_column_of_column():
    assert list(compute(t['name']['name'], data)) == \
            list(compute(t['name'], data))


def test_Distinct():
    assert set(compute(Distinct(t['name']), data)) == set(['Alice', 'Bob'])
    assert set(compute(Distinct(t), data)) == set(map(tuple, data))


def test_Distinct_count():
    t2 = t['name'].distinct()
    gby = by(t2, t2['name'], t2['name'].count())
    result = set(compute(gby, data))
    assert result == set([('Alice', 1), ('Bob', 1)])


def test_sort():
    assert list(compute(t.sort('amount'), data)) == \
            sorted(data, key=lambda x: x[1], reverse=False)

    assert list(compute(t.sort('amount', ascending=True), data)) == \
            sorted(data, key=lambda x: x[1], reverse=False)

    assert list(compute(t.sort(['amount', 'id']), data)) == \
            sorted(data, key=lambda x: (x[1], x[2]), reverse=False)


def test_fancy_sort():
    assert list(compute(t.sort(t['amount']), data)) ==\
            list(compute(t.sort('amount'), data))

    assert list(compute(t.sort(t[['amount', 'id']]), data)) ==\
            list(compute(t.sort(['amount', 'id']), data))

    assert list(compute(t.sort(0-t['amount']), data)) ==\
            list(compute(t.sort('amount'), data))[::-1]


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

    t_idx = TableSymbol('t_idx', '{name: string, b: int32}')
    t_arc = TableSymbol('t_arc', '{a: int32, b: int32}')
    t_wanted = TableSymbol('t_wanted', '{name: string}')

    j = join(join(t_idx, t_arc, 'b'), t_wanted, 'name')[['name', 'b', 'a']]

    result = compute(j, {t_idx: idx, t_arc: arc, t_wanted: wanted})
    result = sorted(map(tuple, result))
    expected = sorted([('A', 3, 1),
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
    names = TableSymbol('names', '{first: string, last: string}')

    siblings = join(names.relabel({'first': 'left'}),
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
    assert list(compute(t['amount'].map(inc), data)) == [x[1] + 1 for x in data]


def test_map():
    assert (list(compute(t.map(lambda _, amt, id: amt + id), data)) ==
            [x[1] + x[2] for x in data])


def test_apply_column():
    result = compute(Apply(builtins.sum, t['amount']), data)
    expected = compute(t['amount'].sum(), data)

    assert result == expected


def test_apply():
    data2 = tuple(map(tuple, data))
    assert compute(Apply(hash, t), data2) == hash(data2)


def test_map_datetime():
    from datetime import datetime
    data = [['A', 0], ['B', 1]]
    t = TableSymbol('t', '{foo: string, datetime: int64}')

    result = list(compute(t['datetime'].map(datetime.utcfromtimestamp), data))
    expected = [datetime(1970, 1, 1, 0, 0, 0), datetime(1970, 1, 1, 0, 0, 1)]

    assert result == expected


def test_by_multi_column_grouper():
    t = TableSymbol('t', '{x: int, y: int, z: int}')
    expr = by(t, t[['x', 'y']], t['z'].count())
    data = [(1, 2, 0), (1, 2, 0), (1, 1, 0)]

    print(set(compute(expr, data)))
    assert set(compute(expr, data)) == set([(1, 2, 2), (1, 1, 1)])


def test_merge():
    col = (t['amount'] * 2).label('new')

    expr = merge(t['name'], col)

    assert list(compute(expr, data)) == [(row[0], row[1] * 2) for row in data]


def test_map_columnwise():
    colwise = t['amount'] * t['id']

    expr = colwise.map(lambda x: x / 10, schema="{mod: int64}", iscolumn=True)

    assert list(compute(expr, data)) == [((row[1]*row[2]) / 10) for row in data]


def test_map_columnwise_of_selection():
    tsel = t[t['name'] == 'Alice']
    colwise = tsel['amount'] * tsel['id']

    expr = colwise.map(lambda x: x / 10, schema="{mod: int64}", iscolumn=True)

    assert list(compute(expr, data)) == [((row[1]*row[2]) / 10) for row in data[::2]]


def test_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]

    assert list(compute(expr, data)) == ['Alice']


def test_recursive_rowfunc():
    f = rrowfunc(t['name'], t)
    assert [f(row) for row in data] == [row[0] for row in data]

    f = rrowfunc(t['amount'] + t['id'], t)
    assert [f(row) for row in data] == [row[1] + row[2] for row in data]

    assert raises(Exception, lambda: rrowfunc(t[t['amount'] < 0]['name'], t))


def test_recursive_rowfunc_is_used():
    expr = by(t, t['name'], (2 * (t['amount'] + t['id'])).sum())
    expected = [('Alice', 2*(101 + 53)),
                ('Bob', 2*(202))]
    assert set(compute(expr, data)) == set(expected)


class TestFunctionExpressions(object):
    def test_compound(self):
        s = t.amount.mean()
        r = compute(s, data)
        assert isinstance(r, float)

        expr = cos(s) ** 2 + sin(s) ** 2
        result = compute(expr, data)
        expected = math.cos(r) ** 2 + math.sin(r) ** 2
        assert result == expected

    def test_user_defined_function(self):
        s = t.amount.count()
        r = compute(s, data)
        assert isinstance(r, int)

        def myfunc(x):
            return (cos(x) + sin(x)) ** 2 / math.pi

        result = compute(myfunc(s), data)
        expected = (math.cos(r) + math.sin(r)) ** 2 / math.pi
        assert result == expected

    def test_user_defined_calls(self):
        s = t.amount.count()
        r = compute(s, data)

        def myother(y):
            return 2 + y ** 10

        def myfunc(x):
            return myother((cos(x) + sin(x)) ** 2 / math.pi)

        result = compute(myfunc(s), data)
        expected = myother((math.cos(r) + math.sin(r)) ** 2 / math.pi)
        assert result == expected


def test_union():
    L1 = [['Alice', 100, 1],
          ['Bob', 200, 2],
          ['Alice', 50, 3]]
    L2 = [['Alice', 100, 4],
          ['Bob', 200, 5],
          ['Alice', 50, 6]]
    L3 = [['Alice', 100, 7],
          ['Bob', 200, 8],
          ['Alice', 50, 9]]

    t1 = TableSymbol('t1', '{name: string, amount: int, id: int}')
    t2 = TableSymbol('t2', '{name: string, amount: int, id: int}')
    t3 = TableSymbol('t3', '{name: string, amount: int, id: int}')

    expr = union(t1, t2, t3)

    result = list(compute(expr, {t1: L1, t2: L2, t3: L3}))

    assert result == L1 + L2 + L3


def test_by_groupby_deep():
    data = [(1, 2, 'Alice'),
            (1, 3, 'Bob'),
            (2, 4, 'Alice'),
            (2, 4, '')]

    schema = '{x: int, y: int, name: string}'
    t = TableSymbol('t', schema)

    t2 = t[t['name'] != '']
    t3 = merge(t2.x, t2.name)
    expr = by(t3, t3.name, t3.x.mean())
    result = set(compute(expr, data))
    assert result == set([('Alice', 1.5), ('Bob', 1.0)])


def test_by_then_sort_dict_items_sequence():
    expr = by(tbig, tbig.name, tbig.amount.sum()).sort('name')
    assert compute(expr, databig)


def test_summary():
    expr = summary(count=t.id.count(), sum=t.amount.sum())
    assert compute(expr, data) == (3, 350)


def test_summary_by():
    expr = by(t, t.name, summary(count=t.id.count(), sum=t.amount.sum()))
    assert set(compute(expr, data)) == set([('Alice', 2, 150),
                                            ('Bob', 1, 200)])

    expr = by(t, t.name, summary(count=t.id.count(), sum=(t.amount + 1).sum()))
    assert set(compute(expr, data)) == set([('Alice', 2, 152),
                                            ('Bob', 1, 201)])

    expr = by(t, t.name, summary(count=t.id.count(), sum=t.amount.sum() + 1))
    assert set(compute(expr, data)) == set([('Alice', 2, 151),
                                            ('Bob', 1, 201)])


def test_reduction_arithmetic():
    expr = t.amount.sum() + 1
    assert compute(expr, data) == 351


def test_scalar_arithmetic():
    x = ScalarSymbol('x', 'real')
    y = ScalarSymbol('y', 'real')
    assert compute(x + y, {x: 2, y: 3}) == 5
    assert compute_one(x + y, 2, 3) == 5
    assert compute_one(x * y, 2, 3) == 6
    assert compute_one(x / y, 6, 3) == 2
    assert compute_one(x % y, 4, 3) == 1
    assert compute_one(x ** y, 4, 3) == 64

    assert compute(x + 1, {x: 2}) == 3
    assert compute(x * 2, {x: 2}) == 4
    assert compute(1 + x, {x: 2}) == 3
    assert compute(2 * x, {x: 2}) == 4

    assert compute_one(-x, 1) == -1

    assert compute_one(scalar.numbers.sin(x), 1) == math.sin(1)
