from __future__ import absolute_import, print_function, division

import sys
import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info.major == 3,
    reason="PyHive doesn't work with Python 3.x"
)

pyspark = pytest.importorskip('pyspark')
py4j = pytest.importorskip('py4j')
sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('pyhive.sqlalchemy_hive')

import os
import itertools
import shutil

from functools import partial

from py4j.protocol import Py4JJavaError
import numpy as np
import pandas as pd
import pandas.util.testing as tm
from blaze import compute, symbol, into, by, sin, exp, cos, tan, join
from blaze.interactive import iscorescalar

from pyspark.sql import DataFrame as SparkDataFrame

try:
    from pyspark.sql.utils import AnalysisException
except ImportError:
    AnalysisException = Py4JJavaError

from pyspark import HiveContext
from pyspark.sql import Row
from odo import odo, discover
from odo.utils import tmpfile


data = [['Alice', 100.0, 1],
        ['Bob', 200.0, 2],
        ['Alice', 50.0, 3]]

date_data = []

np.random.seed(0)

for attr in ('YearBegin', 'MonthBegin', 'Day', 'Hour', 'Minute', 'Second'):
    rng = pd.date_range(start='now', periods=len(data),
                        freq=getattr(pd.datetools, attr)()).values
    date_data += list(zip(np.random.choice(['Alice', 'Bob', 'Joe', 'Lester'],
                                           size=len(data)),
                          np.random.rand(len(data)) * 100,
                          np.random.randint(100, size=3),
                          rng))


cities_data = [['Alice', 'NYC'],
               ['Bob', 'Boston']]

df = pd.DataFrame(data, columns=['name', 'amount', 'id'])
date_df = pd.DataFrame(date_data, columns=['name', 'amount', 'id', 'ds'])
cities_df = pd.DataFrame(cities_data, columns=['name', 'city'])


# sc is from conftest.py


@pytest.yield_fixture(scope='module')
def sql(sc):
    try:
        yield HiveContext(sc)
    finally:
        dbpath = 'metastore_db'
        logpath = 'derby.log'
        if os.path.exists(dbpath):
            assert os.path.isdir(dbpath), '%s is not a directory' % dbpath
            shutil.rmtree(dbpath)
        if os.path.exists(logpath):
            assert os.path.isfile(logpath), '%s is not a file' % logpath
            os.remove(logpath)


@pytest.yield_fixture(scope='module')
def people(sc):
    with tmpfile('.txt') as fn:
        df.to_csv(fn, header=False, index=False)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        yield parts.map(
            lambda person: Row(
                name=person[0], amount=float(person[1]), id=int(person[2])
            )
        )


@pytest.yield_fixture(scope='module')
def cities(sc):
    with tmpfile('.txt') as fn:
        cities_df.to_csv(fn, header=False, index=False)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        yield parts.map(lambda person: Row(name=person[0], city=person[1]))


@pytest.yield_fixture(scope='module')
def date_people(sc):
    with tmpfile('.txt') as fn:
        date_df.to_csv(fn, header=False, index=False)
        raw = sc.textFile(fn)
        parts = raw.map(lambda line: line.split(','))
        yield parts.map(
            lambda person: Row(
                name=person[0],
                amount=float(person[1]),
                id=int(person[2]),
                ds=pd.Timestamp(person[3]).to_pydatetime()
            )
        )


@pytest.fixture(scope='module')
def ctx(sql, people, cities, date_people):
    sql.registerDataFrameAsTable(sql.createDataFrame(people), 't')
    sql.cacheTable('t')
    sql.registerDataFrameAsTable(sql.createDataFrame(cities), 's')
    sql.cacheTable('s')
    sql.registerDataFrameAsTable(sql.createDataFrame(date_people), 'dates')
    sql.cacheTable('dates')
    return sql


@pytest.fixture(scope='module')
def db(ctx):
    return symbol('db', discover(ctx))


def test_projection(db, ctx):
    expr = db.t[['id', 'name']]
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result) == into(set, expected)


def test_symbol_compute(db, ctx):
    assert isinstance(compute(db.t, ctx, return_type='native'), SparkDataFrame)


def test_field_access(db, ctx):
    for field in db.t.fields:
        expr = getattr(db.t, field)
        result = into(pd.Series, compute(expr, ctx, return_type='native'))
        expected = compute(expr, {db: {'t': df}}, return_type='native')
        assert result.name == expected.name
        np.testing.assert_array_equal(result.values,
                                      expected.values)


def test_head(db, ctx):
    expr = db.t[['name', 'amount']].head(2)
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(list, result) == into(list, expected)


def test_literals(db, ctx):
    expr = db.t[db.t.amount >= 100]
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert list(map(set, into(list, result))) == list(
        map(set, into(list, expected))
    )


def test_by_summary(db, ctx):
    t = db.t
    expr = by(t.name, mymin=t.amount.min(), mymax=t.amount.max())
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result) == into(set, expected)


def test_join(db, ctx):
    expr = join(db.t, db.s)
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df, 's': cities_df}}, return_type='native')

    assert isinstance(result, SparkDataFrame)
    assert into(set, result) == into(set, expected)
    assert discover(result) == expr.dshape


def test_join_diff_contexts(db, ctx, cities):
    expr = join(db.t, db.s, 'name')
    people = ctx.table('t')
    cities = into(ctx, cities, dshape=discover(ctx.table('s')))
    scope = {db: {'t': people, 's': cities}}
    result = compute(expr, scope, return_type='native')
    expected = compute(expr, {db: {'t': df, 's': cities_df}}, return_type='native')
    assert set(map(frozenset, odo(result, set))) == set(
        map(frozenset, odo(expected, set))
    )


def test_field_distinct(ctx, db):
    expr = db.t.name.distinct()
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


def test_boolean(ctx, db):
    expr = db.t.amount > 50
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


def test_selection(ctx, db):
    expr = db.t[db.t.amount > 50]
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert list(map(set, into(list, result))) == list(
        map(set, into(list, expected))
    )


def test_selection_field(ctx, db):
    expr = db.t[db.t.amount > 50].name
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


@pytest.mark.parametrize(
    ['field', 'reduction'],
    itertools.product(
        ['id', 'amount'],
        ['sum', 'max', 'min', 'mean', 'count', 'nunique']
    )
)
def test_reductions(ctx, db, field, reduction):
    expr = getattr(db.t[field], reduction)()
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(list, result)[0][0] == expected


def test_column_arithmetic(ctx, db):
    expr = db.t.amount + 1
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


@pytest.mark.parametrize('func', [sin, cos, tan, exp])
def test_math(ctx, db, func):
    expr = func(db.t.amount)
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    np.testing.assert_allclose(
        np.sort(odo(result, np.ndarray, dshape=expr.dshape)),
        np.sort(odo(expected, np.ndarray))
    )


@pytest.mark.parametrize(['field', 'ascending'],
                         itertools.product(['name', 'id', ['name', 'amount']],
                                           [True, False]))
def test_sort(ctx, db, field, ascending):
    expr = db.t.sort(field, ascending=ascending)
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert list(map(set, into(list, result))) == list(
        map(set, into(list, expected))
    )


@pytest.mark.xfail
def test_map(ctx, db):
    expr = db.t.id.map(lambda x: x + 1, 'int')
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert into(set, result, dshape=expr.dshape) == into(set, expected)


@pytest.mark.parametrize(
    ['grouper', 'reducer', 'reduction'],
    itertools.chain(
        itertools.product(
            ['name', 'id', ['id', 'amount']],
            ['id', 'amount'],
            ['sum', 'count', 'max', 'min', 'mean', 'nunique']
        ),
        [('name', 'name', 'count'), ('name', 'name', 'nunique')]
    )
)
def test_by(ctx, db, grouper, reducer, reduction):
    t = db.t
    expr = by(t[grouper], total=getattr(t[reducer], reduction)())
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert set(map(frozenset, into(list, result))) == set(
        map(frozenset, into(list, expected))
    )


@pytest.mark.parametrize(
    ['reducer', 'reduction'],
    itertools.product(['id', 'name'], ['count', 'nunique'])
)
def test_multikey_by(ctx, db, reducer, reduction):
    t = db.t
    expr = by(t[['id', 'amount']], total=getattr(t[reducer], reduction)())
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert (set(map(frozenset, into(list, result))) ==
            set(map(frozenset, into(list, expected))))


def test_grouper_with_arith(ctx, db):
    expr = by(db.t[['id', 'amount']], total=(db.t.amount + 1).sum())
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert list(map(set, into(list, result))) == list(map(set, into(list, expected)))


def test_by_non_native_ops(ctx, db):
    expr = by(db.t.id, total=db.t.id.nunique())
    result = compute(expr, ctx, return_type='native')
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert list(map(set, into(list, result))) == list(map(set, into(list, expected)))


def test_str_len(ctx, db):
    expr = db.t.name.str_len()
    result = odo(compute(expr, ctx, return_type='native'), pd.Series)
    expected = compute(expr, {db: {'t': df}}, return_type='native')
    assert result.name == 'name'
    assert expected.name == 'name'
    assert odo(result, set) == odo(expected, set)


@pytest.mark.parametrize(
    'attr',
    ['year', 'month', 'day', 'hour', 'minute', 'second'] + list(
        map(
            partial(
                pytest.mark.xfail,
                raises=(Py4JJavaError, AnalysisException)
            ),
            ['millisecond', 'microsecond']
        )
    )
)
def test_by_with_date(ctx, db, attr):
    # TODO: investigate CSV writing precision between pandas 0.16.0 and 0.16.1
    # TODO: see if we can use odo to convert the dshape of an existing
    #       DataFrame
    expr = by(getattr(db.dates.ds, attr), mean=db.dates.amount.mean())
    result = odo(
        compute(expr, ctx, return_type='native'), pd.DataFrame
    ).sort('mean').reset_index(drop=True)
    expected = compute(
        expr,
        {db: {'dates': date_df}},
        return_type='native'
    ).sort('mean').reset_index(drop=True)
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize('keys', [[1], [1, 2]])
def test_isin(ctx, db, keys):
    expr = db.t[db.t.id.isin(keys)]
    result = odo(compute(expr, ctx, return_type='native'), set)
    expected = odo(compute(expr, {db: {'t': df}}, return_type='native'), set)
    assert (set(map(frozenset, odo(result, list))) ==
            set(map(frozenset, odo(expected, list))))


def test_nunique_spark_dataframe(ctx, db):
    result = odo(compute(db.t.nunique(), ctx, return_type='native'), int)
    expected = ctx.table('t').distinct().count()
    assert result == expected


def test_core_compute(ctx, db):
    assert isinstance(compute(db.t, ctx, return_type='core'), pd.DataFrame)
    assert isinstance(compute(db.t.amount, ctx, return_type='core'), pd.Series)
    assert iscorescalar(compute(db.t.amount.mean(), ctx, return_type='core'))
    assert isinstance(compute(db.t, ctx, return_type=list), list)
