from __future__ import absolute_import, division, print_function

import pytest
pyspark = pytest.importorskip('pyspark')

import pandas as pd
from blaze import compute, symbol, summary, exp, by, join, merge
from toolz import identity


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]

data2 = [['Alice', 'Austin'],
         ['Bob', 'Boston']]

df = pd.DataFrame(data, columns=['name', 'amount', 'id'])


# this only exists because we need to have a single session scoped spark
# context, otherwise these would simply be global variables
@pytest.fixture
def rdd(sc):
    return sc.parallelize(data)


@pytest.fixture
def rdd2(sc):
    return sc.parallelize(data2)


t = symbol('t', 'var * {name: string, amount: int, id: int}')

t2 = symbol('t2', 'var * {name: string, city: string}')

# Web Commons Graph Example data
data_idx = [['A', 1],
            ['B', 2],
            ['C', 3]]

data_arc = [[1, 3],
            [2, 3],
            [3, 1]]

t_idx = symbol('idx', 'var * {name: string, node_id: int32}')

t_arc = symbol('arc', 'var * {node_out: int32, node_id: int32}')


def test_symbol(rdd):
    assert compute(t, rdd) == rdd


def test_projection(rdd):
    assert compute(t['name'], rdd).collect() == [row[0] for row in data]


def test_multicols_projection(rdd):
    result = compute(t[['amount', 'name']], rdd).collect()
    expected = [(100, 'Alice'), (200, 'Bob'), (50, 'Alice')]

    print(result)
    print(expected)

    assert result == expected


reduction_exprs = [
    t['amount'].sum(),
    t['amount'].min(),
    t['amount'].max(),
    t['amount'].nunique(),
    t['name'].nunique(),
    t['amount'].count(),
    (t['amount'] > 150).any(),
    (t['amount'] > 150).all(),
    t['amount'].mean(),
    t['amount'].var(),
    summary(a=t.amount.sum(), b=t.id.count()),
    t['amount'].std()
]


def test_reductions(rdd):
    for expr in reduction_exprs:
        result = compute(expr, rdd)
        expected = compute(expr, data)
        if not result == expected:
            print(result)
            print(expected)
            if isinstance(result, float):
                assert abs(result - expected) < 0.001
            else:
                assert result == expected

exprs = [
    t['amount'],
    t['amount'] == 100,
    t['amount'].truncate(150),
    t[t['name'] == 'Alice'],
    t[t['amount'] == 0],
    t[t['amount'] > 150],
    t['amount'] + t['id'],
    t['amount'] % t['id'],
    exp(t['amount']),
    by(t['name'], total=t['amount'].sum()),
    by(t['name'], total=(t['amount'] + 1).sum()),
    (t['amount'] * 1).label('foo'),
    t.map(lambda tup: tup[1] + tup[2], 'real'),
    t[t.name.like('Alice')],
    t['amount'].apply(identity, 'var * real', splittable=True),
    t['amount'].map(lambda x: x + 1, 'int')
]

exprs = list(zip(map(str, exprs), exprs))


def tuplify(x):
    return tuple(x) if isinstance(x, list) else x


@pytest.mark.parametrize(['string', 'expr'], exprs)
def test_basic(rdd, string, expr):
    result = set(map(tuplify, compute(expr, rdd).collect()))
    expected = set(map(tuplify, compute(expr, data)))
    assert result == expected


tbig = symbol(
    'tbig', 'var * {name: string, sex: string[1], amount: int, id: int}')

big_exprs = [
    by(tbig[['name', 'sex']], total=tbig['amount'].sum()),
    by(tbig[['name', 'sex']], total=(tbig['id'] + tbig['amount']).sum())]


@pytest.mark.parametrize('expr', big_exprs)
def test_big_by(sc, expr):
    data = [['Alice', 'F', 100, 1],
            ['Alice', 'F', 100, 3],
            ['Drew', 'F', 100, 4],
            ['Drew', 'M', 100, 5],
            ['Drew', 'M', 200, 5]]
    rdd = sc.parallelize(data)
    result = set(map(tuplify, compute(expr, rdd).collect()))
    expected = set(map(tuplify, compute(expr, data)))
    assert result == expected


def test_head(rdd):
    assert list(compute(t.head(1), rdd)) == list(compute(t.head(1), data))


sort_exprs = [
    t.sort('amount'),
    t.sort('amount', ascending=True),
    t.sort(t['amount'], ascending=True),
    t.sort(-t['amount'].label('foo') + 1, ascending=True),
    t.sort(['amount', 'id'])
]


@pytest.mark.parametrize('expr', sort_exprs)
def test_sort(rdd, expr):
    result = compute(expr, rdd).collect()
    expected = list(compute(expr, data))
    assert result == expected


def test_distinct(rdd):
    assert set(compute(t['name'].distinct(), rdd).collect()) == \
        set(['Alice', 'Bob'])


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason='cannot specify columns to distinct on yet',
)
def test_distinct_on(rdd):
    compute(t.distinct('name'), rdd)


def test_join(rdd, rdd2):

    joined = join(t, t2, 'name')
    expected = [('Alice', 100, 1, 'Austin'),
                ('Bob', 200, 2, 'Boston'),
                ('Alice', 50, 3, 'Austin')]
    result = compute(joined, {t: rdd, t2: rdd2}).collect()
    assert all(i in expected for i in result)


def test_multi_column_join(sc):
    left = [(1, 2, 3),
            (2, 3, 4),
            (1, 3, 5)]
    right = [(1, 2, 30),
             (1, 3, 50),
             (1, 3, 150)]
    rleft = sc.parallelize(left)
    rright = sc.parallelize(right)

    L = symbol('L', 'var * {x: int, y: int, z: int}')
    R = symbol('R', 'var * {x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    result = compute(j, {L: rleft, R: rright})
    expected = [(1, 2, 3, 30),
                (1, 3, 5, 50),
                (1, 3, 5, 150)]

    assert set(result.collect()) == set(expected)


def test_groupby(sc):
    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = join(t_arc, t_idx, "node_id")

    t = by(joined['name'], count=joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx: rddidx})
    in_degree = dict(a.collect())
    assert in_degree == {'A': 1, 'C': 2}


def test_multi_level_rowfunc_works(rdd):
    expr = t['amount'].map(lambda x: x + 1, 'int')

    assert compute(expr, rdd).collect() == [x[1] + 1 for x in data]


def test_merge(rdd):
    col = (t['amount'] * 2).label('new')
    expr = merge(t['name'], col)

    assert compute(expr, rdd).collect() == [
        (row[0], row[1] * 2) for row in data]


def test_selection_out_of_order(rdd):
    expr = t['name'][t['amount'] < 100]

    assert compute(expr, rdd).collect() == ['Alice']


def test_recursive_rowfunc_is_used(rdd):
    expr = by(t['name'], total=(2 * (t['amount'] + t['id'])).sum())
    expected = [('Alice', 2 * (101 + 53)),
                ('Bob', 2 * (202))]
    assert set(compute(expr, rdd).collect()) == set(expected)


def test_outer_join(sc):
    left = [(1, 'Alice', 100),
            (2, 'Bob', 200),
            (4, 'Dennis', 400)]
    left = sc.parallelize(left)
    right = [('NYC', 1),
             ('Boston', 1),
             ('LA', 3),
             ('Moscow', 4)]
    right = sc.parallelize(right)

    L = symbol('L', 'var * {id: int, name: string, amount: real}')
    R = symbol('R', 'var * {city: string, id: int}')

    assert set(compute(join(L, R), {L: left, R: right}).collect()) == set(
        [(1, 'Alice', 100, 'NYC'),
         (1, 'Alice', 100, 'Boston'),
         (4, 'Dennis', 400, 'Moscow')])

    assert set(compute(join(L, R, how='left'), {L: left, R: right}).collect()) == set(
        [(1, 'Alice', 100, 'NYC'),
         (1, 'Alice', 100, 'Boston'),
         (2, 'Bob', 200, None),
         (4, 'Dennis', 400, 'Moscow')])

    assert set(compute(join(L, R, how='right'), {L: left, R: right}).collect()) == set(
        [(1, 'Alice', 100, 'NYC'),
         (1, 'Alice', 100, 'Boston'),
         (3, None, None, 'LA'),
         (4, 'Dennis', 400, 'Moscow')])

    # Full outer join not yet supported
    assert set(compute(join(L, R, how='outer'), {L: left, R: right}).collect()) == set(
        [(1, 'Alice', 100, 'NYC'),
         (1, 'Alice', 100, 'Boston'),
         (2, 'Bob', 200, None),
         (3, None, None, 'LA'),
         (4, 'Dennis', 400, 'Moscow')])
