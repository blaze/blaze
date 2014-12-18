from __future__ import absolute_import, division, print_function

from into import into
from blaze.compute import compute, compute_up
from blaze.compatibility import xfail
from blaze.expr import *
from blaze.expr.functions import *
from toolz import identity
from datashape.predicates import iscollection
from datashape import dshape

import pytest


data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]
data2 = [['Alice', 'Austin'],
         ['Bob', 'Boston']]

from pandas import DataFrame
df = DataFrame(data, columns=['name', 'amount', 'id'])

pyspark = pytest.importorskip('pyspark')
pytest.importorskip('pyspark.sql')

from pyspark import RDD
sc = pyspark.SparkContext("local", "Simple App")
rdd = sc.parallelize(data)
rdd2 = sc.parallelize(data2)


t = symbol('t', 'var * {name: string, amount: int, id: int}')

t2 = symbol('t2', 'var * {name: string, city: string}')

#Web Commons Graph Example data
data_idx = [['A', 1],
            ['B', 2],
            ['C', 3]]

data_arc = [[1, 3],
            [2, 3],
            [3, 1]]

t_idx = symbol('idx', 'var * {name: string, node_id: int32}')

t_arc = symbol('arc', 'var * {node_out: int32, node_id: int32}')

def test_spark_symbol():
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
    summary(a=t.amount.sum(), b=t.id.count()),
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
    t.like(name='Alice'),
    t['amount'].apply(identity, 'var * real', splittable=True),
    t['amount'].map(inc, 'int')]


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
    tbig = symbol('tbig', 'var * {name: string, sex: string[1], amount: int, id: int}')

    big_exprs = [
        by(tbig[['name', 'sex']], total=tbig['amount'].sum()),
        by(tbig[['name', 'sex']], total=(tbig['id'] + tbig['amount']).sum())]

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
    expected = [('Alice', 100, 1, 'Austin'),
                ('Bob', 200, 2, 'Boston'),
                ('Alice', 50, 3, 'Austin')]
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

    L = symbol('L', 'var * {x: int, y: int, z: int}')
    R = symbol('R', 'var * {x: int, y: int, w: int}')

    j = join(L, R, ['x', 'y'])

    result = compute(j, {L: rleft, R: rright})
    expected = [(1, 2, 3, 30),
                (1, 3, 5, 50),
                (1, 3, 5, 150)]

    print(result.collect())
    assert set(result.collect()) ==  set(expected)


def test_spark_groupby():
    rddidx = sc.parallelize(data_idx)
    rddarc = sc.parallelize(data_arc)

    joined = join(t_arc, t_idx, "node_id")

    result_blaze = compute(joined, {t_arc: rddarc, t_idx:rddidx})
    t = by(joined['name'], count=joined['node_id'].count())
    a = compute(t, {t_arc: rddarc, t_idx:rddidx})
    in_degree = dict(a.collect())
    assert in_degree == {'A': 1, 'C': 2}


def test_spark_multi_level_rowfunc_works():
    expr = t['amount'].map(lambda x: x + 1, 'int')

    assert compute(expr, rdd).collect() == [x[1] + 1 for x in data]


@xfail(reason="pandas-numexpr-platform doesn't play well with spark")
def test_spark_merge():
    col = (t['amount'] * 2).label('new')
    expr = merge(t['name'], col)

    assert compute(expr, rdd).collect() == [(row[0], row[1] * 2) for row in data]


def test_spark_into():
    seq = [1, 2, 3]
    assert isinstance(into(rdd, seq), RDD)
    assert into([], into(rdd, seq)) == seq


def test_spark_selection_out_of_order():
    expr = t['name'][t['amount'] < 100]

    assert compute(expr, rdd).collect() == ['Alice']


def test_spark_recursive_rowfunc_is_used():
    expr = by(t['name'], total=(2 * (t['amount'] + t['id'])).sum())
    expected = [('Alice', 2*(101 + 53)),
                ('Bob', 2*(202))]
    assert set(compute(expr, rdd).collect()) == set(expected)


def test_spark_outer_join():
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
    """
    assert set(compute(join(L, R, how='outer'), {L: left, R: right}).collect()) == set(
            [(1, 'Alice', 100, 'NYC'),
             (1, 'Alice', 100, 'Boston'),
             (2, 'Bob', 200, None),
             (3, None, None, 'LA'),
             (4, 'Dennis', 400, 'Moscow')])
    """

def test_discover():
    assert discover(rdd).subshape[0] == discover(data).subshape[0]

### SparkSQL

from pyspark.sql import SchemaRDD, SQLContext

if issubclass(SQLContext, object):
    sqlContext = SQLContext(sc)


    def test_into_SparkSQL_from_PySpark():
        srdd = into(sqlContext, data, schema=t.schema)
        assert isinstance(srdd, SchemaRDD)

        assert into(list, rdd) == into(list, srdd)

    def test_into_sparksql_from_other():
        srdd = into(sqlContext, df)
        assert isinstance(srdd, SchemaRDD)
        assert into(list, srdd) == into(list, df)


    def test_SparkSQL_discover():
        srdd = into(sqlContext, data, schema=t.schema)
        assert discover(srdd).subshape[0] == \
                dshape('{name: string, amount: int64, id: int64}')


    def test_sparksql_compute():
        srdd = into(sqlContext, data, schema=t.schema)
        assert compute_up(t, srdd).context == sqlContext
        assert discover(compute_up(t, srdd).query).subshape[0] == \
                dshape('{name: string, amount: int64, id: int64}')

        assert isinstance(compute(t[['name', 'amount']], srdd),
                          SchemaRDD)

        assert sorted(compute(t.name, srdd).collect()) == ['Alice', 'Alice', 'Bob']

        assert isinstance(compute(t[['name', 'amount']].head(2), srdd),
                         (tuple, list))


    def test_sparksql_with_literals():
        srdd = into(sqlContext, data, schema=t.schema)
        expr = t[t.amount >= 100]
        result = compute(expr, srdd)
        assert isinstance(result, SchemaRDD)
        assert set(map(tuple, result.collect())) == \
                set(map(tuple, compute(expr, data)))


    def test_sparksql_by_summary():
        t = symbol('t', 'var * {name: string, amount: int64, id: int64}')
        srdd = into(sqlContext, data, schema=t.schema)
        expr = by(t.name, mymin=t.amount.min(), mymax=t.amount.max())
        result = compute(expr, srdd)
        assert result.collect()
        assert (str(discover(result)).replace('?', '')
             == str(expr.dshape).replace('?', ''))


    def test_spqrksql_join():
        accounts = symbol('accounts', 'var * {name: string, amount: int64, id: int64}')
        accounts_rdd = into(sqlContext, data, schema=accounts.schema)

        cities = symbol('cities', 'var * {name: string, city: string}')
        cities_data = [('Alice', 'NYC'), ('Bob', 'LA')]
        cities_rdd = into(sqlContext,
                          cities_data,
                          schema='{name: string, city: string}')

        expr = join(accounts, cities)

        result = compute(expr, {cities: cities_rdd, accounts: accounts_rdd})

        assert isinstance(result, SchemaRDD)

        assert (str(discover(result)).replace('?', '') ==
                str(expr.dshape))

    def test_comprehensive():
        L = [[100, 1, 'Alice'],
             [200, 2, 'Bob'],
             [300, 3, 'Charlie'],
             [400, 4, 'Dan'],
             [500, 5, 'Edith']]

        df = DataFrame(L, columns=['amount', 'id', 'name'])

        rdd = into(sc, df)
        srdd = into(sqlContext, df)

        t = symbol('t', 'var * {amount: int64, id: int64, name: string}')

        expressions = {
                t: [],
                t['id']: [],
                t.id.max(): [],
                t.amount.sum(): [],
                t.amount + 1: [],
                sin(t.amount): [srdd], # sparksql without hiveql doesn't support math
                exp(t.amount): [srdd], # sparksql without hiveql doesn't support math
                t.amount > 50: [],
                t[t.amount > 50]: [],
                t.sort('name'): [],
                t.sort('name', ascending=False): [],
                t.head(3): [],
                t.name.distinct(): [],
                t[t.amount > 50]['name']: [],
                t.id.map(lambda x: x + 1, 'int'): [srdd], # no udfs yet
                t[t.amount > 50]['name']: [],
                by(t.name, total=t.amount.sum()): [],
                by(t.id, total=t.id.count()): [],
                by(t[['id', 'amount']], total=t.id.count()): [],
                by(t[['id', 'amount']], total=(t.amount + 1).sum()): [],
                by(t[['id', 'amount']], total=t.name.nunique()): [rdd, srdd],
                by(t.id, total=t.amount.count()): [],
                by(t.id, total=t.id.nunique()): [rdd, srdd],
                # by(t, t.count()): [],
                # by(t.id, t.count()): [df],
                t[['amount', 'id']]: [],
                t[['id', 'amount']]: [],
                }

        for e, exclusions in expressions.items():
            if rdd not in exclusions:
                if iscollection(e.dshape):
                    assert into(set, compute(e, rdd)) == into(set, compute(e, df))
                else:
                    assert compute(e, rdd) == compute(e, df)
            if srdd not in exclusions:
                if iscollection(e.dshape):
                    assert into(set, compute(e, srdd)) == into(set, compute(e, df))
                else:
                    assert compute(e, rdd) == compute(e, df)
