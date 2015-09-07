from datetime import timedelta
import itertools
import re

import pytest

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

import numpy as np
import pandas as pd

import pandas.util.testing as tm

from odo import odo, resource, drop, discover
from blaze import symbol, compute, concat, join


names = ('tbl%d' % i for i in itertools.count())


def normalize(s):
    s = ' '.join(s.strip().split()).lower()
    s = re.sub(r'(alias)_?\d*', r'\1', s)
    return re.sub(r'__([A-Za-z_][A-Za-z_0-9]*)', r'\1', s)


@pytest.fixture
def url():
    return 'postgresql://postgres@localhost/test::%s'


@pytest.yield_fixture
def sql(url):
    try:
        t = resource(url % next(names), dshape='var * {A: string, B: int64}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([('a', 1), ('b', 2)], t)
        try:
            yield t
        finally:
            drop(t)


@pytest.yield_fixture
def sqla(url):
    try:
        t = resource(url % next(names), dshape='var * {A: ?string, B: ?int32}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([('a', 1), (None, 1), ('c', None)], t)
        try:
            yield t
        finally:
            drop(t)


@pytest.yield_fixture
def sqlb(url):
    try:
        t = resource(url % next(names), dshape='var * {A: string, B: int64}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([('a', 1), ('b', 2)], t)
        try:
            yield t
        finally:
            drop(t)


@pytest.yield_fixture
def sql_with_dts(url):
    try:
        t = resource(url % next(names), dshape='var * {A: datetime}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([(d,) for d in pd.date_range('2014-01-01', '2014-02-01')], t)
        try:
            yield t
        finally:
            drop(t)


@pytest.yield_fixture
def sql_two_tables(url):
    dshape = 'var * {a: int32}'
    try:
        t = resource(url % next(names), dshape=dshape)
        u = resource(url % next(names), dshape=dshape)
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield u, t
        finally:
            drop(t)
            drop(u)


@pytest.yield_fixture
def sql_with_float(url):
    try:
        t = resource(url % next(names), dshape='var * {c: float64}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield t
        finally:
            drop(t)


def test_postgres_create(sql):
    assert odo(sql, list) == [('a', 1), ('b', 2)]


def test_postgres_isnan(sql_with_float):
    data = (1.0,), (float('nan'),)
    table = odo(data, sql_with_float)
    sym = symbol('s', discover(data))
    assert odo(compute(sym.isnan(), table), list) == [(False,), (True,)]


def test_insert_from_subselect(sql_with_float):
    data = pd.DataFrame([{'c': 2.0}, {'c': 1.0}])
    tbl = odo(data, sql_with_float)
    s = symbol('s', discover(data))
    odo(compute(s[s.c.isin((1.0, 2.0))].sort(), tbl), sql_with_float),
    tm.assert_frame_equal(
        odo(sql_with_float, pd.DataFrame).iloc[2:].reset_index(drop=True),
        pd.DataFrame([{'c': 1.0}, {'c': 2.0}]),
    )


def test_concat(sql_two_tables):
    t_table, u_table = sql_two_tables
    t_data = pd.DataFrame(np.arange(5), columns=['a'])
    u_data = pd.DataFrame(np.arange(5, 10), columns=['a'])
    odo(t_data, t_table)
    odo(u_data, u_table)

    t = symbol('t', discover(t_data))
    u = symbol('u', discover(u_data))
    tm.assert_frame_equal(
        odo(
            compute(concat(t, u).sort('a'), {t: t_table, u: u_table}),
            pd.DataFrame,
        ),
        pd.DataFrame(np.arange(10), columns=['a']),
    )


def test_concat_invalid_axis(sql_two_tables):
    t_table, u_table = sql_two_tables
    t_data = pd.DataFrame(np.arange(5), columns=['a'])
    u_data = pd.DataFrame(np.arange(5, 10), columns=['a'])
    odo(t_data, t_table)
    odo(u_data, u_table)

    # We need to force the shape to not be a record here so we can
    # create the `Concat` node with an axis=1.
    t = symbol('t', '5 * 1 * int32')
    u = symbol('u', '5 * 1 * int32')

    with pytest.raises(ValueError) as e:
        compute(concat(t, u, axis=1), {t: t_table, u: u_table})

    # Preserve the suggestion to use merge.
    assert "'merge'" in str(e.value)


def test_timedelta_arith(sql_with_dts):
    delta = timedelta(days=1)
    dates = pd.Series(pd.date_range('2014-01-01', '2014-02-01'))
    sym = symbol('s', discover(dates))
    assert (
        odo(compute(sym + delta, sql_with_dts), pd.Series) == dates + delta
    ).all()
    assert (
        odo(compute(sym - delta, sql_with_dts), pd.Series) == dates - delta
    ).all()


def test_coerce_bool_and_sum(sql):
    n = sql.name
    t = symbol(n, discover(sql))
    expr = (t.B > 1.0).coerce(to='int32').sum()
    result = compute(expr, sql).scalar()
    expected = odo(compute(t.B, sql), pd.Series).gt(1).sum()
    assert result == expected


def test_distinct_on(sql):
    t = symbol('t', discover(sql))
    computation = compute(t[['A', 'B']].sort('A').distinct('A'), sql)
    assert normalize(str(computation)) == normalize("""
    SELECT DISTINCT ON (anon_1."A") anon_1."A", anon_1."B"
    FROM (SELECT {tbl}."A" AS "A", {tbl}."B" AS "B"
    FROM {tbl}) AS anon_1 ORDER BY anon_1."A" ASC
    """.format(tbl=sql.name))
    assert odo(computation, tuple) == (('a', 1), ('b', 2))


def test_join_type_promotion(sqla, sqlb):
    t, s = symbol(sqla.name, discover(sqla)), symbol(sqlb.name, discover(sqlb))
    expr = join(t, s, 'B', how='inner')
    result = set(map(tuple, compute(expr, {t: sqla, s: sqlb}).execute().fetchall()))
    expected = set([(1, 'a', 'a'), (1, None, 'a')])
    assert result == expected
