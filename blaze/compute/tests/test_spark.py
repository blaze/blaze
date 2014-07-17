from __future__ import absolute_import, division, print_function

from blaze.compute.spark import *
from blaze.compatibility import xfail
from blaze.expr.table import *

import pytest


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]
data2 = [['Alice', 'Austin'],
         ['Bob', 'Boston']]


pyspark = pytest.importorskip('pyspark')
sc = pyspark.SparkContext("local", "Simple App")
rdd = sc.parallelize(data)
rdd2 = sc.parallelize(data2)


t = TableSymbol('t', '{name: string, amount: int, id: int}')

t2 = TableSymbol('t2', '{name: string, city: string}')

#Web Commons Graph Example data
data_idx = [['A', 1],
            ['B', 2],
            ['C', 3]]

data_arc = [[1, 3],
            [2, 3],
            [3, 1]]

t_idx = TableSymbol('idx', '{name: string, node_id: int32}')

t_arc = TableSymbol('arc', '{node_out: int32, node_id: int32}')

def test_spark_table():
    assert compute(t, rdd) == rdd


def test_spark_projection():
    assert compute(t['name'], rdd).collect() == [row[0] for row in data]


def test_spark_multicols_projection():
    result = compute(t[['amount', 'name']], rdd).collect()
    expected = [(100, 'Alice'), (200, 'Bob'), (50, 'Alice')]

    print(result)
    print(expected)

    assert result == expected


inc = lambda x: x + 1


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
    t['amount'].std()]


def test_spark_reductions():
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
    t[t['name'] == 'Alice'],
    t[t['amount'] == 0],
    t[t['amount'] > 150],
    t['amount'] + t['id'],
    t['amount'] % t['id'],
    exp(t['amount']),
    by(t, t['name'], t['amount'].sum()),
    by(t, t['name'], (t['amount'] + 1).sum()),
    (t['amount'] * 1).label('foo'),
    t.map(lambda _, amt, id: amt + id),
    t['amount'].map(inc)]


def test_spark_basic():
    check_exprs_against_python(exprs, data, rdd)


def check_exprs_against_python(exprs, data, rdd):
    any_bad = False
    for expr in exprs:
        result = compute(expr, rdd).collect()
        expected = list(compute(expr, data))
        if not result == expected:
            any_bad = True
            print("Expression:", expr)
            print("Spark:", result)
            print("Python:", expected)

    assert not any_bad


def test_spark_big_by():
    tbig = TableSymbol('tbig', '{name: string, sex: string[1], amount: int, id: int}')

    big_exprs = [
        by(tbig, tbig[['name', 'sex']], tbig['amount'].sum()),
        by(tbig, tbig[['name', 'sex']], (tbig['id'] + tbig['amount']).sum())]

    databig = [['Alice', 'F', 100, 1],
               ['Alice', 'F', 100, 3],
               ['Drew', 'F', 100, 4],
               ['Drew', 'M', 100, 5],
               ['Drew', 'M', 200, 5]]

    rddbig = sc.parallelize(databig)

    check_exprs_against_python(big_exprs, databig, rddbig)


def test_spark_head():
    assert list(compute(t.head(1), rdd)) == list(compute(t.head(1), data))


def test_spark_sort():
    check_exprs_against_python([
                t.sort('amount'),
                t.sort('amount', ascending=True),
                t.sort(t['amount'], ascending=True),
                t.sort(-t['amount'].label('foo') + 1, ascending=True),
                t.sort(['amount', 'id'])], data, rdd)

def test_spark_distinct():
    assert set(compute(t['name'].distinct(), rdd).collect()) == \
            set(['Alice', 'Bob'])



def test_spark_join():

    joined = join(t, t2, 'name')
    expected = [['Alice', 100, 1, 'Austin'],
                ['Bob', 200, 2, 'Boston'],
                ['Alice', 50, 3, 'Austin']]
    result = compute(joined, {t: rdd, t2: rdd2}).collect()
    assert all(i in expected for i in result)


def test_spark_multi_column_join():
    left = [(1, 2, 3),
            (2, 3, 4),
            (1, 3, 5)]
    right = [(1, 2, 30),
             (1, 3, 50),
             (1, 3, 150)]
    rleft = sc.parallelize(left)
    rright = sc.parallelize(right)

    L = TableSymbol('L', '{x: int, y: int, z: int}')
    R = TableSymbol('R', '{x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    result = compute(j, {L: rleft, R: rright})
    expected = [(1, 2, 3, 30),
                (1, 3, 5, 50),
                (1, 3, 5, 150)]

    print(result.collect())
    assert result.collect() == expected


def test_spark_groupby():
    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = join(t_arc, t_idx, "node_id")

    result_blaze = compute(joined, {t_arc: rddarc, t_idx:rddidx})
    t = by(joined, joined['name'], joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx:rddidx})
    in_degree = dict(a.collect())
    assert in_degree == {'A': 1, 'C': 2}


def test_spark_multi_level_rowfunc_works():
    expr = t['amount'].map(lambda x: x + 1)

    assert compute(expr, rdd).collect() == [x[1] + 1 for x in data]


@xfail(reason="pandas-numexpr-platform doesn't play well with spark")
def test_spark_merge():
    col = (t['amount'] * 2).label('new')
    expr = merge(t['name'], col)

    assert compute(expr, rdd).collect() == [(row[0], row[1] * 2) for row in data]


def test_spark_into():
    from blaze.api.into import into
    seq = [1, 2, 3]
    assert isinstance(into(rdd, seq), RDD)
    assert into([], into(rdd, seq)) == seq


def test_spark_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]

    assert compute(expr, rdd).collect() == ['Alice']


def test_spark_recursive_rowfunc_is_used():
    expr = by(t, t['name'], (2 * (t['amount'] + t['id'])).sum())
    expected = [('Alice', 2*(101 + 53)),
                ('Bob', 2*(202))]
    assert set(compute(expr, rdd).collect()) == set(expected)
