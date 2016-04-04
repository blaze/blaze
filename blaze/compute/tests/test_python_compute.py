from __future__ import absolute_import, division, print_function
from collections import Mapping

import math
import itertools
import operator
import pytest
from datetime import datetime, date
import datashape
from datashape.py2help import mappingproxy
from collections import Iterator, Iterable

import blaze
from blaze.compute.python import (nunique, mean, rrowfunc, rowfunc,
                                  reduce_by_funcs, optimize)
from blaze import dshape, symbol, discover
from blaze.compatibility import PY2
from blaze.compute.core import compute, compute_up, pre_compute
from blaze.expr import (by, merge, join, distinct, sum, min, max, any, summary,
                        count, std, head, sample, transform, greatest, least)
import numpy as np

from blaze import cos, sin
from blaze.compatibility import builtins
from blaze.utils import raises


t = symbol('t', 'var * {name: string, amount: int, id: int}')


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]


tbig = symbol('tbig', 'var * {name: string, sex: string[1], amount: int, id: int}')


databig = [['Alice', 'F', 100, 1],
           ['Alice', 'F', 100, 3],
           ['Drew', 'F', 100, 4],
           ['Drew', 'M', 100, 5],
           ['Drew', 'M', 200, 5]]


def test_dispatched_rowfunc():
    cw = optimize(t['amount'] + 100, [])
    assert rowfunc(t)(t) == t
    assert rowfunc(cw)(('Alice', 100, 1)) == 200


def test_reduce_by_funcs():
    e = summary(number=t.id.max(), sum=t.amount.sum())
    b = by(t, e)
    assert reduce_by_funcs(b)[2]([1,2,3], [4,5,6]) == (1, 7)


def test_symbol():
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
    expr = optimize(-t.amount, [])
    assert list(compute(expr, data)) == [-x[1] for x in data]


def test_reductions():
    assert compute(sum(t['amount']), data) == 100 + 200 + 50
    assert compute(min(t['amount']), data) == 50
    assert compute(max(t['amount']), data) == 200
    assert compute(nunique(t['amount']), data) == 3
    assert compute(nunique(t['name']), data) == 2
    assert compute(count(t['amount']), data) == 3
    assert compute(any(t['amount'] > 150), data) is True
    assert compute(any(t['amount'] > 250), data) is False
    assert compute(t.amount[0], data) == 100
    assert compute(t.amount[-1], data) == 50


def test_1d_reductions_keepdims():
    for r in [sum, min, max, nunique, count]:
        assert compute(r(t.amount, keepdims=True), data) == \
               (compute(r(t.amount), data),)


def test_count():
    t = symbol('t', '3 * int')
    assert compute(t.count(), [1, None, 2]) == 2


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


def test_std():
    amt = [row[1] for row in data]
    assert np.allclose(compute(t.amount.std(), data), np.std(amt))
    assert np.allclose(compute(t.amount.std(unbiased=True), data),
                       np.std(amt, ddof=1))
    assert np.allclose(compute(t.amount.var(), data), np.var(amt))
    assert np.allclose(compute(t.amount.var(unbiased=True), data),
                       np.var(amt, ddof=1))


def test_by_no_grouper():
    names = t['name']
    assert set(compute(by(names, count=names.count()), data)) == \
            set([('Alice', 2), ('Bob', 1)])


def test_by_one():
    print(compute(by(t['name'], total=t['amount'].sum()), data))
    assert set(compute(by(t['name'], total=t['amount'].sum()), data)) == \
            set([('Alice', 150), ('Bob', 200)])


def test_by_compound_apply():
    print(compute(by(t['name'], total=(t['amount'] + 1).sum()), data))
    assert set(compute(by(t['name'], total=(t['amount'] + 1).sum()), data)) == \
            set([('Alice', 152), ('Bob', 201)])


def test_by_two():
    result = compute(by(tbig[['name', 'sex']], total=tbig['amount'].sum()),
                     databig)

    expected = [('Alice', 'F', 200),
                ('Drew', 'F', 100),
                ('Drew', 'M', 300)]

    print(set(result))
    assert set(result) == set(expected)


def test_by_three():
    result = compute(by(tbig[['name', 'sex']],
                        total=(tbig['id'] + tbig['amount']).sum()),
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
    left = [['Alice', 100], ['Bob', 200]]
    right = [['Alice', 1], ['Bob', 2]]

    L = symbol('L', 'var * {name: string, amount: int}')
    R = symbol('R', 'var * {name: string, id: int}')
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

    L = symbol('L', 'var * {id: int, name: string, amount: real}')
    R = symbol('R', 'var * {city: string, id: int}')

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

    L = symbol('L', 'var * {x: int, y: int, z: int}')
    R = symbol('R', 'var * {x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    print(list(compute(j, {L: left, R: right})))
    assert list(compute(j, {L: left, R: right})) == [(1, 2, 3, 30),
                                                     (1, 3, 5, 50),
                                                     (1, 3, 5, 150)]


@pytest.mark.xfail(reason="This doesn't necessarily make sense")
def test_column_of_column():
    assert list(compute(t['name']['name'], data)) == \
            list(compute(t['name'], data))


def test_distinct():
    assert set(compute(distinct(t['name']), data)) == set(['Alice', 'Bob'])
    assert set(compute(distinct(t), data)) == set(map(tuple, data))
    e = distinct(t)
    assert list(compute(e, [])) == []


def test_distinct_count():
    t2 = t['name'].distinct()
    gby = by(t2, total=t2.count())
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


def test_sort_on_column():
    assert list(compute(t.name.distinct().sort('name'), data)) == \
            ['Alice', 'Bob']


def test_head():
    assert list(compute(t.head(1), data)) == [data[0]]

    e = head(t, 101)
    p = list(range(1000))
    assert len(list(compute(e, p))) == 101


def test_sample():
    NN = len(databig)
    for n in range(1, NN+1):
        assert (len(compute(tbig.sample(n=n), databig)) ==
                len(compute(tbig.sample(frac=float(n)/NN), databig)) ==
                n)
    assert len(compute(tbig.sample(n=NN*2), databig)) == NN


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

    t_idx = symbol('t_idx', 'var * {name: string, b: int32}')
    t_arc = symbol('t_arc', 'var * {a: int32, b: int32}')
    t_wanted = symbol('t_wanted', 'var * {name: string}')

    # >>> compute(join(t_idx, t_arc, 'b'), {t_idx: idx, t_arc: arc})
    # [[1, A, 3],
    #  [1, A, 2],
    #  [1, A, 5],
    #  [3, C, 1],
    #  [3, C, 2],
    #  [3, C, 4],
    #  [3, C, 5],
    #  [6, F, 1],
    #  [6, F, 2],
    #  [6, F, 4]]

    j = join(join(t_idx, t_arc, 'b'), t_wanted, 'name')[['name', 'b', 'a']]

    result = compute(j, {t_idx: idx, t_arc: arc, t_wanted: wanted})
    result = sorted(map(tuple, result))
    expected = sorted([('A', 1, 3),
                       ('A', 1, 2),
                       ('A', 1, 5),
                       ('F', 6, 1),
                       ('F', 6, 2),
                       ('F', 6, 4)])

    assert result == expected


def test_label():
    assert list(compute((t['amount'] * 1).label('foo'), data)) == \
            list(compute((t['amount'] * 1), data))


def test_relabel_join():
    names = symbol('names', 'var * {first: string, last: string}')

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
    assert list(compute(t['amount'].map(inc, 'int'), data)) == [x[1] + 1 for x in data]


def test_map():
    assert (list(compute(t.map(lambda tup: tup[1] + tup[2], 'int'), data)) ==
            [x[1] + x[2] for x in data])


def test_apply_column():
    result = compute(t.amount.apply(builtins.sum, 'real'), data)
    expected = compute(t.amount.sum(), data)

    assert result == expected


def test_apply():
    data2 = tuple(map(tuple, data))
    assert compute(t.apply(hash, 'int'), data2) == hash(data2)


def test_map_datetime():
    from datetime import datetime
    data = [['A', 0], ['B', 1]]
    t = symbol('t', 'var * {foo: string, datetime: int64}')

    result = list(compute(t['datetime'].map(datetime.utcfromtimestamp,
    'datetime'), data))
    expected = [datetime(1970, 1, 1, 0, 0, 0), datetime(1970, 1, 1, 0, 0, 1)]

    assert result == expected


def test_by_multi_column_grouper():
    t = symbol('t', 'var * {x: int, y: int, z: int}')
    expr = by(t[['x', 'y']], total=t['z'].count())
    data = [(1, 2, 0), (1, 2, 0), (1, 1, 0)]

    print(set(compute(expr, data)))
    assert set(compute(expr, data)) == set([(1, 2, 2), (1, 1, 1)])


def test_merge():
    col = (t['amount'] * 2).label('new')

    expr = merge(t['name'], col)

    assert list(compute(expr, data)) == [(row[0], row[1] * 2) for row in data]


def test_transform():
    expr = transform(t, x=t.amount / t.id)
    assert list(compute(expr, data)) == [('Alice', 100, 1, 100),
                                         ('Bob',   200, 2, 100),
                                         ('Alice',  50, 3, 50 / 3)]


def test_map_columnwise():
    colwise = t['amount'] * t['id']

    expr = colwise.map(lambda x: x / 10, 'int64', name='mod')

    assert list(compute(expr, data)) == [((row[1]*row[2]) / 10) for row in data]


def test_map_columnwise_of_selection():
    tsel = t[t['name'] == 'Alice']
    colwise = tsel['amount'] * tsel['id']

    expr = colwise.map(lambda x: x / 10, 'int64', name='mod')

    assert list(compute(expr, data)) == [((row[1]*row[2]) / 10) for row in data[::2]]


def test_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]

    assert list(compute(expr, data)) == ['Alice']


def test_recursive_rowfunc():
    f = rrowfunc(t['name'], t)
    assert [f(row) for row in data] == [row[0] for row in data]

    expr = optimize(t['amount'] + t['id'], [])
    f = rrowfunc(expr, t)
    assert [f(row) for row in data] == [row[1] + row[2] for row in data]

    assert raises(Exception, lambda: rrowfunc(t[t['amount'] < 0]['name'], t))


def test_recursive_rowfunc_is_used():
    expr = by(t['name'], total=(2 * (t['amount'] + t['id'])).sum())
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


def test_by_groupby_deep():
    data = [(1, 2, 'Alice'),
            (1, 3, 'Bob'),
            (2, 4, 'Alice'),
            (2, 4, '')]

    schema = '{x: int, y: int, name: string}'
    t = symbol('t', datashape.var * schema)

    t2 = t[t['name'] != '']
    t3 = merge(t2.x, t2.name)
    expr = by(t3.name, avg=t3.x.mean())
    result = set(compute(expr, data))
    assert result == set([('Alice', 1.5), ('Bob', 1.0)])


def test_by_then_sort_dict_items_sequence():
    expr = by(tbig.name, total=tbig.amount.sum()).sort('name')
    assert compute(expr, databig)


def test_summary():
    expr = summary(count=t.id.count(), sum=t.amount.sum())
    assert compute(expr, data) == (3, 350)
    assert compute(expr, iter(data)) == (3, 350)


def test_summary_keepdims():
    assert compute(summary(count=t.id.count(), sum=t.amount.sum(),
                           keepdims=True), data) == \
          (compute(summary(count=t.id.count(), sum=t.amount.sum(),
                           keepdims=False), data),)


def test_summary_by():
    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum()))
    assert set(compute(expr, data)) == set([('Alice', 2, 150),
                                            ('Bob', 1, 200)])

    expr = by(t.name, summary(count=t.id.count(), sum=(t.amount + 1).sum()))
    assert set(compute(expr, data)) == set([('Alice', 2, 152),
                                            ('Bob', 1, 201)])

    expr = by(t.name, summary(count=t.id.count(), sum=t.amount.sum() + 1))
    assert set(compute(expr, data)) == set([('Alice', 2, 151),
                                            ('Bob', 1, 201)])


def test_summary_by_first():
    expr = by(t.name, amt=t.amount[0])
    assert set(compute(expr, data)) == set((('Bob', 200), ('Alice', 100)))


def test_summary_by_last():
    expr = by(t.name, amt=t.amount[-1])
    assert set(compute(expr, data)) == set((('Bob', 200), ('Alice', 50)))


def test_reduction_arithmetic():
    expr = t.amount.sum() + 1
    assert compute(expr, data) == 351


def test_scalar_arithmetic():
    x = symbol('x', 'real')
    y = symbol('y', 'real')
    assert compute(x + y, {x: 2, y: 3}) == 5
    assert compute_up(x + y, 2, 3) == 5
    assert compute_up(x * y, 2, 3) == 6
    assert compute_up(x / y, 6, 3) == 2
    assert compute_up(x % y, 4, 3) == 1
    assert compute_up(x ** y, 4, 3) == 64

    assert compute(x + 1, {x: 2}) == 3
    assert compute(x * 2, {x: 2}) == 4
    assert compute(1 + x, {x: 2}) == 3
    assert compute(2 * x, {x: 2}) == 4

    assert compute_up(-x, 1) == -1

    assert compute_up(blaze.sin(x), 1) == math.sin(1)


def test_like():
    t = symbol('t', 'var * {name: string, city: string}')
    data = [
        ('Alice Smith', 'New York'),
        ('Bob Smith', 'Chicago'),
        ('Alice Walker', 'LA')
    ]

    assert list(compute(t[t.name.like('Alice*')], data)) == [data[0], data[2]]
    assert list(compute(t[t.name.like('lice*')], data)) == []
    assert list(compute(t[t.name.like('*Smith*')], data)) == [data[0], data[1]]
    assert list(
        compute(t[t.name.like('*Smith*') & t.city.like('New York')], data)
    ) == [data[0]]


def test_datetime_comparison():
    data = [['Alice', date(2000, 1, 1)],
            ['Bob', date(2000, 2, 2)],
            ['Alice', date(2000, 3, 3)]]

    t = symbol('t', 'var * {name: string, when: date}')

    assert list(compute(t[t.when > '2000-01-01'], data)) == data[1:]


def test_datetime_access():
    data = [['Alice', 100, 1, datetime(2000, 1, 1, 1, 1, 1)],
            ['Bob', 200, 2, datetime(2000, 1, 1, 1, 1, 1)],
            ['Alice', 50, 3, datetime(2000, 1, 1, 1, 1, 1)]]

    t = symbol('t',
            'var * {amount: float64, id: int64, name: string, when: datetime}')

    assert list(compute(t.when.year, data)) == [2000, 2000, 2000]
    assert list(compute(t.when.second, data)) == [1, 1, 1]
    assert list(compute(t.when.date, data)) == [date(2000, 1, 1)] * 3


def test_utcfromtimestamp():
    t = symbol('t', '1 * int64')
    assert list(compute(t.utcfromtimestamp, [0])) == \
            [datetime(1970, 1, 1, 0, 0)]


payments = [{'name': 'Alice', 'payments': [
                {'amount':  100, 'when': datetime(2000, 1, 1, 1, 1 ,1)},
                {'amount':  200, 'when': datetime(2000, 2, 2, 2, 2, 2)}
                ]},
            {'name': 'Bob', 'payments': [
                {'amount':  300, 'when': datetime(2000, 3, 3, 3, 3 ,3)},
                {'amount': -400, 'when': datetime(2000, 4, 4, 4, 4, 4)},
                {'amount':  500, 'when': datetime(2000, 5, 5, 5, 5, 5)}
                ]},
            ]

payments_ordered = [('Alice', [( 100, datetime(2000, 1, 1, 1, 1 ,1)),
                               ( 200, datetime(2000, 2, 2, 2, 2, 2))]),
                    ('Bob',   [( 300, datetime(2000, 3, 3, 3, 3 ,3)),
                               (-400, datetime(2000, 4, 4, 4, 4, 4)),
                               ( 500, datetime(2000, 5, 5, 5, 5, 5))])]

payment_dshape = 'var * {name: string, payments: var * {amount: int32, when: datetime}}'


@pytest.mark.xfail(reason="Can't reason about nested broadcasts yet")
def test_nested():
    t = symbol('t', payment_dshape)
    assert list(compute(t.name, payments_ordered)) == ['Alice', 'Bob']

    assert list(compute(t.payments, payments_ordered)) == \
                [p[1] for p in payments_ordered]
    assert list(compute(t.payments.amount, payments_ordered)) == \
            [(100, 200), (300, -400, 500)]
    assert list(compute(t.payments.amount + 1, payments_ordered)) ==\
            [(101, 201), (301, -399, 501)]


@pytest.mark.xfail(reason="Can't reason about nested broadcasts yet")
def test_scalar():
    s = symbol('s', '{name: string, id: int32, payments: var * {amount: int32, when: datetime}}')
    data = ('Alice', 1, ((100, datetime(2000, 1, 1, 1, 1 ,1)),
                         (200, datetime(2000, 2, 2, 2, 2, 2)),
                         (300, datetime(2000, 3, 3, 3, 3, 3))))

    assert compute(s.name, data) == 'Alice'
    assert compute(s.id + 1, data) == 2
    assert tuple(compute(s.payments.amount, data)) == (100, 200, 300)
    assert tuple(compute(s.payments.amount + 1, data)) == (101, 201, 301)


def test_slice():
    assert compute(t[0], data) == data[0]
    assert list(compute(t[:2], data)) == list(data[:2])
    assert list(compute(t.name[:2], data)) == [data[0][0], data[1][0]]


def test_negative_slicing():
    assert list(compute(t[-1:], data)) == data[-1:]
    assert list(compute(t[-1:], iter(data))) == data[-1:]
    assert list(compute(t[-1], data)) == data[-1]
    assert list(compute(t[-1], iter(data))) == data[-1]
    assert list(compute(t[-2], data)) == data[-2]
    assert list(compute(t[-2], iter(data))) == data[-2]


@pytest.mark.xfail(raises=ValueError,
                   reason="No support for stop and step having negative values")
def test_negative_slicing_raises_on_stop_and_step_not_None():
    assert list(compute(t[-2:-5:-1], data)) == data[-2:-5:-1]


def test_multi_dataset_broadcast():
    x = symbol('x', '3 * int')
    y = symbol('y', '3 * int')

    a = [1, 2, 3]
    b = [10, 20, 30]

    assert list(compute(x + y, {x: a, y: b})) == [11, 22, 33]
    assert list(compute(2*x + (y + 1), {x: a, y: b})) == [13, 25, 37]


@pytest.mark.xfail(reason="Optimize doesn't create multi-table-broadcasts")
def test_multi_dataset_broadcast_with_Record_types():
    x = symbol('x', '3 * {p: int, q: int}')
    y = symbol('y', '3 * int')

    a = [(1, 1), (2, 2), (3, 3)]
    b = [10, 20, 30]

    assert list(compute(x.p + x.q + y, {x: iter(a), y: iter(b)})) == [12, 24, 36]


def eq(a, b):
    if isinstance(a, (Iterable, Iterator)):
        a = list(a)
    if isinstance(b, (Iterable, Iterator)):
        b = list(b)
    return a == b


def test_pre_compute():
    s = symbol('s', 'var * {a: int, b: int}')
    assert pre_compute(s, [(1, 2)]) == [(1, 2)]
    assert list(pre_compute(s, iter([(1, 2)]))) == [(1, 2)]
    assert list(pre_compute(s, iter([(1, 2), (3, 4)]))) == [(1, 2), (3, 4)]
    assert list(pre_compute(s, iter([{'a': 1, 'b': 2},
                                     {'a': 3, 'b': 4}]))) == [(1, 2), (3, 4)]


def test_dicts():
    t = symbol('t', 'var * {name: string, amount: int, id: int}')

    L = [['Alice', 100, 1],
         ['Bob', 200, 2],
         ['Alice', 50, 3]]

    d = [{'name': 'Alice', 'amount': 100, 'id': 1},
         {'name': 'Bob', 'amount': 200, 'id': 2},
         {'name': 'Alice', 'amount': 50, 'id': 3}]

    assert list(pre_compute(t, d)) == list(map(tuple, L))

    for expr in [t.amount, t.amount.sum(), by(t.name, sum=t.amount.sum())]:
        assert eq(compute(expr, {t: L}),
                  compute(expr, {t: d}))

    for expr in [t.amount, t.amount.sum(), by(t.name, sum=t.amount.sum())]:
        assert eq(compute(expr, {t: iter(L)}),
                  compute(expr, {t: iter(d)}))
        assert eq(compute(expr, {t: iter(L)}),
                  compute(expr, {t: L}))


def test_nelements_list_tuple():
    assert compute(t.nelements(), data) == len(data)


def test_nelements_iterator():
    x = (row for row in data)
    assert compute(t.nelements(), x) == len(data)


def test_nrows():
    assert compute(t.nrows, data) == len(data)
    x = (row for row in data)
    assert compute(t.nrows, x) == len(data)


@pytest.mark.xfail(raises=Exception, reason="Only 1D reductions allowed")
def test_nelements_2D():
    assert compute(t.nelements(axis=1), data) == len(data[0])


def test_compute_field_on_dicts():
    s = symbol('s', '{x: 3 * int, y: 3 * int}')
    d = {'x': [1, 2, 3], 'y': [4, 5, 6]}
    assert compute(s.x, {s: d}) == [1, 2, 3]


class _MapProxy(Mapping):  # pragma: no cover
    def __init__(self, mapping):
        self._map = mapping

    def __getitem__(self, key):
        return self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)


@pytest.mark.parametrize(
    'cls',
    # mappingproxy refers to PyDictProxy in py2 which is not registered
    # as a mapping
    (_MapProxy,) + (mappingproxy,) if not PY2 else (),
)
def test_comute_on_mapping(cls):
    s = symbol('s', '{x: 3 * int64, y: 3 * float32}')
    d = cls({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]})
    assert compute(s.x, {s: d}, return_type='native') == [1, 2, 3]


def test_truncate():
    s = symbol('x', 'real')
    assert compute(s.truncate(20), 154) == 140
    assert compute(s.truncate(0.1), 3.1415) == 3.1


def test_truncate_datetime():
    s = symbol('x', 'datetime')
    assert compute(s.truncate(2, 'days'), datetime(2002, 1, 3, 12, 30)) ==\
            date(2002, 1, 2)

    s = symbol('x', 'var * datetime')
    assert list(compute(s.truncate(2, 'days'),
                        [datetime(2002, 1, 3, 12, 30)])) ==\
            [date(2002, 1, 2)]


def test_compute_up_on_base():
    d = datetime.now()
    s = symbol('s', 'datetime')
    assert compute(s.minute, d) == d.minute


def test_notnull():
    data = [('Alice', -100, None),
            (None, None, None),
            ('Bob', 300, 'New York City')]
    t = symbol('t', 'var * {name: ?string, amount: ?int32, city: ?string}')
    expr = t.name.notnull()
    result = compute(expr, data)
    assert list(result) == [True, False, True]


def test_notnull_whole_collection():
    t = symbol('t', 'var * {name: ?string, amount: ?int32, city: ?string}')
    with pytest.raises(AttributeError):
        t.notnull


@pytest.mark.parametrize('keys', [['Alice'], ['Bob', 'Alice']])
def test_isin(keys):
    expr = t[t.name.isin(keys)]
    result = list(compute(expr, data))
    expected = [el for el in data if el[0] in keys]
    assert result == expected


def test_greatest():
    s_data, t_data = [1, 2], [2, 1]
    s, t = symbol('s', discover(s_data)), symbol('t', discover(t_data))
    result = compute(greatest(s, t), {s: s_data, t: t_data})
    expected = np.maximum(s_data, t_data).tolist()
    assert list(result) == expected


def test_least():
    s_data, t_data = [1, 2], [2, 1]
    s, t = symbol('s', discover(s_data)), symbol('t', discover(t_data))
    result = compute(least(s, t), {s: s_data, t: t_data})
    expected = np.minimum(s_data, t_data).tolist()
    assert list(result) == expected
