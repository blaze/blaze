from __future__ import absolute_import, print_function, division

import os
import pytest

xfail = pytest.mark.xfail

pytest.importorskip('pyspark')
pytest.importorskip('pyspark.sql')
sa = pytest.importorskip('sqlalchemy')

import pandas as pd
from datashape.predicates import iscollection
from datashape import dshape
from blaze import discover, compute, symbol, into, by, sin, exp, join
from pyspark.sql import SQLContext, Row, DataFrame as SparkDataFrame
from into.utils import tmpfile


data = [['Alice', 100.0, 1],
        ['Bob', 200.0, 2],
        ['Alice', 50.0, 3]]

df = pd.DataFrame(data, columns=['name', 'amount', 'id'])

# sc is from conftest.py


@pytest.yield_fixture(scope='module')
def ctx(sc):
    sql = SQLContext(sc)
    with tmpfile('.txt') as fn:
        lines = os.linesep.join(','.join(map(str, row)) for row in data)
        lines += os.linesep
        with open(fn, mode='wb') as f:
            f.write(lines)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        people = parts.map(lambda person: Row(name=person[0],
                                              amount=float(person[1]),
                                              id=int(person[2])))
        schema = sql.inferSchema(people)
        schema.registerTempTable('t')
        yield sql


@pytest.fixture(scope='module')
def db(ctx):
    return symbol('db', discover(ctx))


@xfail(reason='moving to into')
def test_into_SparkSQL_from_PySpark(db, ctx):
    srdd = into(ctx, data, schema=db.t.dshape)
    assert isinstance(srdd, SparkDataFrame)

    # assert into(list, rdd) == into(list, srdd)


@xfail(reason='moving to into')
def test_into_sparksql_from_other(ctx):
    srdd = into(ctx, df)
    assert isinstance(srdd, SparkDataFrame)
    assert into(list, srdd) == into(list, df)


@xfail(reason='moving to into')
def test_discover(db, ctx):
    srdd = into(ctx, data, schema=db.t.schema)
    assert discover(srdd).subshape[0] == \
        dshape('{name: string, amount: int64, id: int64}')


def test_projection(db, ctx):
    expr = db.t[['id', 'name']]
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result) == into(set, expected)


def test_symbol_compute(db, ctx):
    assert isinstance(compute(db.t, ctx), SparkDataFrame)


@xfail(raises=AssertionError, reason='detuplify')
def test_field_access(db, ctx):
    expr = db.t.name
    expected = compute(expr, ctx)
    result = compute(expr, {db: {'t': df}})
    assert into(list, expected) == into(list, result)


def test_head(db, ctx):
    expr = db.t[['name', 'amount']].head(2)
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(list, result) == into(list, expected)


def test_literals(db, ctx):
    expr = db.t[db.t.amount >= 100]
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == \
        list(map(set, into(list, expected)))


def test_by_summary(db, ctx):
    t = db.t
    expr = by(t.name, mymin=t.amount.min(), mymax=t.amount.max())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result) == into(set, expected)


@xfail(reason='not worked out yet')
def test_join(db, ctx):
    accounts = symbol(
        'accounts', 'var * {name: string, amount: int64, id: int64}')
    accounts_rdd = into(ctx, data, schema=accounts.schema)

    cities = symbol('cities', 'var * {name: string, city: string}')
    cities_data = [('Alice', 'NYC'), ('Bob', 'LA')]
    cities_rdd = into(ctx,
                      cities_data,
                      schema='{name: string, city: string}')

    expr = join(accounts, cities)

    result = compute(expr, {cities: cities_rdd, accounts: accounts_rdd})

    assert isinstance(result, SparkDataFrame)

    assert (str(discover(result)).replace('?', '') ==
            str(expr.dshape))


@xfail(reason='not worked out yet')
def test_comprehensive(sc, ctx, db):
    L = [[100, 1, 'Alice'],
         [200, 2, 'Bob'],
         [300, 3, 'Charlie'],
         [400, 4, 'Dan'],
         [500, 5, 'Edith']]

    df = pd.DataFrame(L, columns=['amount', 'id', 'name'])

    rdd = into(sc, df)
    srdd = into(ctx, df)

    t = db.t

    expressions = {
        t: [],
        t['id']: [],
        t.id.max(): [],
        t.amount.sum(): [],
        t.amount + 1: [],
        # sparksql without hiveql doesn't support math
        sin(t.amount): [srdd],
        # sparksql without hiveql doesn't support math
        exp(t.amount): [srdd],
        t.amount > 50: [],
        t[t.amount > 50]: [],
        t.sort('name'): [],
        t.sort('name', ascending=False): [],
        t.head(3): [],
        t.name.distinct(): [],
        t[t.amount > 50]['name']: [],
        t.id.map(lambda x: x + 1, 'int'): [srdd],  # no udfs yet
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
                assert into(set, compute(e, rdd)) == into(
                    set, compute(e, df))
            else:
                assert compute(e, rdd) == compute(e, df)
        if srdd not in exclusions:
            if iscollection(e.dshape):
                assert into(set, compute(e, srdd)) == into(
                    set, compute(e, df))
            else:
                assert compute(e, rdd) == compute(e, df)
