from __future__ import absolute_import, print_function, division

import pytest

pyspark = pytest.importorskip('pyspark')
py4j = pytest.importorskip('py4j')
sa = pytest.importorskip('sqlalchemy')

import os
import itertools
import shutil

import numpy as np
import pandas as pd
from blaze import compute, symbol, into, by, sin, exp, cos, tan, join
from blaze.compute.spark import SparkDataFrame
from pyspark import HiveContext, SQLContext
from pyspark.sql import Row, SchemaRDD
from odo import odo, discover
from odo.utils import tmpfile


data = [['Alice', 100.0, 1],
        ['Bob', 200.0, 2],
        ['Alice', 50.0, 3]]

cities_data = [['Alice', 'NYC'],
               ['Bob', 'Boston']]

df = pd.DataFrame(data, columns=['name', 'amount', 'id'])
cities_df = pd.DataFrame(cities_data, columns=['name', 'city'])


# sc is from conftest.py


@pytest.yield_fixture(scope='module')
def sql(sc):
    try:
        if hasattr(pyspark.sql, 'types'):  # pyspark >= 1.3
            yield HiveContext(sc)
        else:
            yield SQLContext(sc)
    finally:
        dbpath = 'metastore_db'
        logpath = 'derby.log'
        if os.path.exists(dbpath):
            assert os.path.isdir(dbpath)
            shutil.rmtree(dbpath)
        if os.path.exists(logpath):
            assert os.path.isfile(logpath)
            os.remove(logpath)


@pytest.yield_fixture(scope='module')
def people(sc):
    with tmpfile('.txt') as fn:
        df.to_csv(fn, header=False, index=False)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        yield parts.map(lambda person: Row(name=person[0],
                                           amount=float(person[1]),
                                           id=int(person[2])))


@pytest.yield_fixture(scope='module')
def cities(sc):
    with tmpfile('.txt') as fn:
        cities_df.to_csv(fn, header=False, index=False)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        yield parts.map(lambda person: Row(name=person[0], city=person[1]))


@pytest.fixture(scope='module')
def ctx(sql, people, cities):
    try:
        sql.registerDataFrameAsTable(sql.createDataFrame(people), 't')
        sql.cacheTable('t')
        sql.registerDataFrameAsTable(sql.createDataFrame(cities), 's')
        sql.cacheTable('s')
    except AttributeError:
        sql.inferSchema(people).registerTempTable('t')
        sql.inferSchema(cities).registerTempTable('s')
    return sql


@pytest.fixture(scope='module')
def db(ctx):
    return symbol('db', discover(ctx))


def test_projection(db, ctx):
    expr = db.t[['id', 'name']]
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result) == into(set, expected)


def test_symbol_compute(db, ctx):
    assert isinstance(compute(db.t, ctx), (SparkDataFrame, SchemaRDD))


def test_field_access(db, ctx):
    for field in db.t.fields:
        expr = getattr(db.t, field)
        result = into(pd.Series, compute(expr, ctx))
        expected = compute(expr, {db: {'t': df}})
        assert result.name == expected.name
        np.testing.assert_array_equal(result.values,
                                      expected.values)


def test_head(db, ctx):
    expr = db.t[['name', 'amount']].head(2)
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(list, result) == into(list, expected)


def test_literals(db, ctx):
    expr = db.t[db.t.amount >= 100]
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == list(map(set, into(list,
                                                                    expected)))


def test_by_summary(db, ctx):
    t = db.t
    expr = by(t.name, mymin=t.amount.min(), mymax=t.amount.max())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result) == into(set, expected)


def test_join(db, ctx):
    expr = join(db.t, db.s)
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df, 's': cities_df}})

    assert isinstance(result, (SparkDataFrame, SchemaRDD))
    assert into(set, result) == into(set, expected)
    assert discover(result) == expr.dshape


def test_join_diff_contexts(db, ctx, cities):
    expr = join(db.t, db.s, 'name')
    people = ctx.table('t')
    cities = into(ctx, cities, dshape=discover(ctx.table('s')))
    scope = {db: {'t': people, 's': cities}}
    result = compute(expr, scope)
    expected = compute(expr, {db: {'t': df, 's': cities_df}})
    assert (set(map(frozenset, odo(result, set))) ==
            set(map(frozenset, odo(expected, set))))


def test_field_distinct(ctx, db):
    expr = db.t.name.distinct()
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


def test_boolean(ctx, db):
    expr = db.t.amount > 50
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


def test_selection(ctx, db):
    expr = db.t[db.t.amount > 50]
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == list(map(set, into(list,
                                                                    expected)))


def test_selection_field(ctx, db):
    expr = db.t[db.t.amount > 50].name
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


@pytest.mark.parametrize(['field', 'reduction'],
                         itertools.product(['id', 'amount'], ['sum', 'max',
                                                              'min', 'mean',
                                                              'count',
                                                              'nunique']))
def test_reductions(ctx, db, field, reduction):
    expr = getattr(db.t[field], reduction)()
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(list, result)[0][0] == expected


def test_column_arithmetic(ctx, db):
    expr = db.t.amount + 1
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


# pyspark doesn't use __version__ so we use this kludge
# should submit a bug report upstream to get __version__
fail_on_spark_one_two = pytest.mark.xfail(not hasattr(pyspark.sql, 'types'),
                                          raises=py4j.protocol.Py4JJavaError,
                                          reason=('math functions only '
                                                  'supported in HiveContext'))


@pytest.mark.parametrize('func', map(fail_on_spark_one_two,
                                     [sin, cos, tan, exp]))
def test_math(ctx, db, func):
    expr = func(db.t.amount)
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    np.testing.assert_allclose(np.sort(odo(result, np.ndarray,
                                           dshape=expr.dshape)),
                               np.sort(odo(expected, np.ndarray)))


@pytest.mark.parametrize(['field', 'ascending'],
                         itertools.product(['name', 'id'], [True, False]))
def test_sort(ctx, db, field, ascending):
    expr = db.t.sort(field, ascending=ascending)
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == list(map(set, into(list,
                                                                    expected)))


@pytest.mark.xfail
def test_map(ctx, db):
    expr = db.t.id.map(lambda x: x + 1, 'int')
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


@pytest.mark.parametrize(['grouper', 'reducer', 'reduction'],
                         itertools.chain(itertools.product(['name', 'id',
                                                            ['id', 'amount']],
                                                           ['id', 'amount'],
                                                           ['sum', 'count',
                                                            'max', 'min',
                                                            'mean',
                                                            'nunique']),
                                         [('name', 'name', 'count'),
                                          ('name', 'name', 'nunique')]))
def test_by(ctx, db, grouper, reducer, reduction):
    t = db.t
    expr = by(t[grouper], total=getattr(t[reducer], reduction)())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert (set(map(frozenset, into(list, result))) ==
            set(map(frozenset, into(list, expected))))


@pytest.mark.parametrize(['reducer', 'reduction'],
                         itertools.product(['id', 'name'],
                                           ['count', 'nunique']))
def test_multikey_by(ctx, db, reducer, reduction):
    t = db.t
    expr = by(t[['id', 'amount']], total=getattr(getattr(t, reducer),
                                                 reduction)())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert (set(map(frozenset, into(list, result))) ==
            set(map(frozenset, into(list, expected))))


def test_grouper_with_arith(ctx, db):
    expr = by(db.t[['id', 'amount']], total=(db.t.amount + 1).sum())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == list(map(set, into(list,
                                                                    expected)))


def test_by_non_native_ops(ctx, db):
    expr = by(db.t.id, total=db.t.id.nunique())
    result = compute(expr, ctx)
    expected = compute(expr, {db: {'t': df}})
    assert list(map(set, into(list, result))) == list(map(set, into(list,
                                                                    expected)))
