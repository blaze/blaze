from datetime import timedelta
from operator import methodcaller
import itertools
import tempfile

import pytest

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

import os
import numpy as np
import pandas as pd

import pandas.util.testing as tm

from datashape import dshape
from odo import odo, resource, drop, discover
from blaze import symbol, compute, concat, by, join, sin, cos, radians, atan2
from odo.utils import tmpfile
from blaze import sqrt, transform, Data
from blaze.interactive import iscorescalar
from blaze.utils import example, normalize


names = ('tbl%d' % i for i in itertools.count())

@pytest.fixture(scope='module')
def pg_ip():
    return os.environ.get('POSTGRES_IP', 'localhost')

@pytest.fixture
def url(pg_ip):
    return 'postgresql://postgres@{}/test::%s'.format(pg_ip)


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


@pytest.yield_fixture(scope='module')
def nyc(pg_ip):
    try:
        t = odo(
            example('nyc.csv'),
            'postgresql://postgres@{}/test::nyc'.format(pg_ip),
        )
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
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
def sql_with_timedeltas(url):
    try:
        t = resource(url % next(names), dshape='var * {N: timedelta}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([(timedelta(seconds=n),) for n in range(10)], t)
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
def products(url):
    try:
        products = resource(url % 'products',
                            dshape="""var * {
                                product_id: int64,
                                color: ?string,
                                price: float64
                            }""", primary_key=['product_id'])
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield products
        finally:
            drop(products)


@pytest.yield_fixture
def orders(url, products):
    try:
        orders = resource(url % 'orders',
                          dshape="""var * {
                            order_id: int64,
                            product_id: map[int64, T],
                            quantity: int64
                          }
                          """,
                          foreign_keys=dict(product_id=products.c.product_id),
                          primary_key=['order_id'])
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield orders
        finally:
            drop(orders)


# TODO: scope these as module because I think pytest is caching sa.Table, which
# doesn't work if remove it after every run

@pytest.yield_fixture
def main(url):
    try:
        main = odo([(i, int(np.random.randint(10))) for i in range(13)],
                   url % 'main',
                   dshape=dshape('var * {id: int64, data: int64}'),
                   primary_key=['id'])
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield main
        finally:
            drop(main)


@pytest.yield_fixture
def pkey(url, main):
    choices = [u'AAPL', u'HPQ', u'ORCL', u'IBM', u'DOW', u'SBUX', u'AMD',
               u'INTC', u'GOOG', u'PRU', u'MSFT', u'AIG', u'TXN', u'DELL',
               u'PEP']
    n = 100
    data = list(zip(range(n),
                    np.random.choice(choices, size=n).tolist(),
                    np.random.uniform(10000, 20000, size=n).tolist(),
                    np.random.randint(main.count().scalar(), size=n).tolist()))
    try:
        pkey = odo(data, url % 'pkey',
                   dshape=dshape('var * {id: int64, sym: string, price: float64, main: map[int64, T]}'),
                   foreign_keys=dict(main=main.c.id),
                   primary_key=['id'])
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield pkey
        finally:
            drop(pkey)


@pytest.yield_fixture
def fkey(url, pkey):
    try:
        fkey = odo([(i,
                     int(np.random.randint(pkey.count().scalar())),
                     int(np.random.randint(10000)))
                    for i in range(10)],
                   url % 'fkey',
                   dshape=dshape('var * {id: int64, sym_id: map[int64, T], size: int64}'),
                   foreign_keys=dict(sym_id=pkey.c.id),
                   primary_key=['id'])
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield fkey
        finally:
            drop(fkey)


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
    assert compute(sym.isnan(), table, return_type=list) == [(False,), (True,)]


def test_insert_from_subselect(sql_with_float):
    data = pd.DataFrame([{'c': 2.0}, {'c': 1.0}])
    tbl = odo(data, sql_with_float)
    s = symbol('s', discover(data))
    odo(compute(s[s.c.isin((1.0, 2.0))].sort(), tbl, return_type='native'), sql_with_float),
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
        compute(concat(t, u).sort('a'), {t: t_table, u: u_table}, return_type=pd.DataFrame),
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
        compute(concat(t, u, axis=1), {t: t_table, u: u_table}, return_type='native')

    # Preserve the suggestion to use merge.
    assert "'merge'" in str(e.value)


def test_timedelta_arith(sql_with_dts):
    delta = timedelta(days=1)
    dates = pd.Series(pd.date_range('2014-01-01', '2014-02-01'))
    sym = symbol('s', discover(dates))
    assert (
        compute(sym + delta, sql_with_dts, return_type=pd.Series) == dates + delta
    ).all()
    assert (
        compute(sym - delta, sql_with_dts, return_type=pd.Series) == dates - delta
    ).all()
    assert (
        compute(sym - (sym - delta), sql_with_dts, return_type=pd.Series) ==
        dates - (dates - delta)
    ).all()


@pytest.mark.parametrize('func', ('var', 'std'))
def test_timedelta_stat_reduction(sql_with_timedeltas, func):
    sym = symbol('s', discover(sql_with_timedeltas))
    expr = getattr(sym.N, func)()

    deltas = pd.Series([timedelta(seconds=n) for n in range(10)])
    expected = timedelta(
        seconds=getattr(deltas.astype('int64') / 1e9, func)(ddof=expr.unbiased)
    )
    assert compute(expr, sql_with_timedeltas, return_type=timedelta) == expected


def test_coerce_bool_and_sum(sql):
    n = sql.name
    t = symbol(n, discover(sql))
    expr = (t.B > 1.0).coerce(to='int32').sum()
    result = compute(expr, sql).scalar()
    expected = compute(t.B, sql, return_type=pd.Series).gt(1).sum()
    assert result == expected


def test_distinct_on(sql):
    t = symbol('t', discover(sql))
    computation = compute(t[['A', 'B']].sort('A').distinct('A'), sql, return_type='native')
    assert normalize(str(computation)) == normalize("""
    SELECT DISTINCT ON (anon_1."A") anon_1."A", anon_1."B"
    FROM (SELECT {tbl}."A" AS "A", {tbl}."B" AS "B"
    FROM {tbl}) AS anon_1 ORDER BY anon_1."A" ASC
    """.format(tbl=sql.name))
    assert odo(computation, tuple) == (('a', 1), ('b', 2))


def test_auto_join_field(orders):
    t = symbol('t', discover(orders))
    expr = t.product_id.color
    result = compute(expr, orders, return_type='native')
    expected = """SELECT
        products.color
    FROM products, orders
    WHERE orders.product_id = products.product_id
    """
    assert normalize(str(result)) == normalize(expected)


def test_auto_join_projection(orders):
    t = symbol('t', discover(orders))
    expr = t.product_id[['color', 'price']]
    result = compute(expr, orders, return_type='native')
    expected = """SELECT
        products.color,
        products.price
    FROM products, orders
    WHERE orders.product_id = products.product_id
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.xfail
@pytest.mark.parametrize('func', ['max', 'min', 'sum'])
def test_foreign_key_reduction(orders, products, func):
    t = symbol('t', discover(orders))
    expr = methodcaller(func)(t.product_id.price)
    result = compute(expr, orders, return_type='native')
    expected = """WITH alias as (select
            products.price as price
        from
            products, orders
        where orders.product_id = products.product_id)
    select {0}(alias.price) as price_{0} from alias
    """.format(func)
    assert normalize(str(result)) == normalize(expected)


def test_foreign_key_chain(fkey):
    t = symbol('t', discover(fkey))
    expr = t.sym_id.main.data
    result = compute(expr, fkey, return_type='native')
    expected = """SELECT
        main.data
    FROM main, fkey, pkey
    WHERE fkey.sym_id = pkey.id and pkey.main = main.id
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.xfail(raises=AssertionError,
                   reason='CTE mucks up generation here')
@pytest.mark.parametrize('grouper', ['sym', ['sym']])
def test_foreign_key_group_by(fkey, grouper):
    t = symbol('fkey', discover(fkey))
    expr = by(t.sym_id[grouper], avg_price=t.sym_id.price.mean())
    result = compute(expr, fkey, return_type='native')
    expected = """SELECT
        pkey.sym,
        avg(pkey.price) AS avg_price
    FROM pkey, fkey
    WHERE fkey.sym_id = pkey.id
    GROUP BY pkey.sym
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.parametrize('grouper', ['sym_id', ['sym_id']])
def test_group_by_map(fkey, grouper):
    t = symbol('fkey', discover(fkey))
    expr = by(t[grouper], id_count=t.size.count())
    result = compute(expr, fkey, return_type='native')
    expected = """SELECT
        fkey.sym_id,
        count(fkey.size) AS id_count
    FROM fkey
    GROUP BY fkey.sym_id
    """
    assert normalize(str(result)) == normalize(expected)


def test_foreign_key_isin(fkey):
    t = symbol('fkey', discover(fkey))
    expr = t.sym_id.isin([1, 2])
    result = compute(expr, fkey, return_type='native')
    expected = """SELECT
        fkey.sym_id IN (%(sym_id_1)s, %(sym_id_2)s) AS anon_1
    FROM fkey
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.xfail(raises=AssertionError, reason='Not yet implemented')
def test_foreign_key_merge_expression(fkey):
    from blaze import merge

    t = symbol('fkey', discover(fkey))
    expr = merge(t.sym_id.sym, t.sym_id.main.data)
    expected = """
        select pkey.sym, main.data
        from
            fkey, pkey, main
        where
            fkey.sym_id = pkey.id and pkey.main = main.id
    """
    result = compute(expr, fkey, return_type='native')
    assert normalize(str(result)) == normalize(expected)


def test_join_type_promotion(sqla, sqlb):
    t, s = symbol(sqla.name, discover(sqla)), symbol(sqlb.name, discover(sqlb))
    expr = join(t, s, 'B', how='inner')
    result = set(map(tuple, compute(expr, {t: sqla, s: sqlb}, return_type='native').execute().fetchall()))
    expected = set([(1, 'a', 'a'), (1, None, 'a')])
    assert result == expected


@pytest.mark.parametrize(['n', 'column'],
                         [(1, 'A'), (-1, 'A'),
                          (1, 'B'), (-1, 'B'),
                          (0, 'A'), (0, 'B')])
def test_shift_on_column(n, column, sql):
    t = symbol('t', discover(sql))
    expr = t[column].shift(n)
    result = compute(expr, sql, return_type=pd.Series)
    expected = odo(sql, pd.DataFrame)[column].shift(n)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize('n', [-1, 0, 1])
def test_shift_arithmetic(sql, n):
    t = symbol('t', discover(sql))
    expr = t.B - t.B.shift(n)
    result = compute(expr, sql, return_type=pd.Series)
    df = odo(sql, pd.DataFrame)
    expected = df.B - df.B.shift(n)
    tm.assert_series_equal(result, expected)


def test_dist(nyc):
    def distance(lat1, lon1, lat2, lon2, R=3959):
        # http://andrew.hedges.name/experiments/haversine/
        dlon = radians(lon2 - lon1)
        dlat = radians(lat2 - lat1)
        a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    t = symbol('t', discover(nyc))

    filtered = t[
        (t.pickup_latitude >= 40.477399) &
        (t.pickup_latitude <= 40.917577) &
        (t.dropoff_latitude >= 40.477399) &
        (t.dropoff_latitude <= 40.917577) &
        (t.pickup_longitude >= -74.259090) &
        (t.pickup_longitude <= -73.700272) &
        (t.dropoff_longitude >= -74.259090) &
        (t.dropoff_longitude <= -73.700272) &
        (t.passenger_count < 6)
    ]
    dist = distance(filtered.pickup_latitude, filtered.pickup_longitude,
                    filtered.dropoff_latitude, filtered.dropoff_longitude)
    transformed = transform(filtered, dist=dist)
    assert (
        compute(transformed.dist.max(), nyc, return_type=float) ==
        compute(transformed.dist, nyc, return_type=pd.Series).max()
    )


def test_multiple_columns_in_transform(nyc):
    t = symbol('t', discover(nyc))
    t = t[
        (t.pickup_latitude >= 40.477399) &
        (t.pickup_latitude <= 40.917577) &
        (t.dropoff_latitude >= 40.477399) &
        (t.dropoff_latitude <= 40.917577) &
        (t.pickup_longitude >= -74.259090) &
        (t.pickup_longitude <= -73.700272) &
        (t.dropoff_longitude >= -74.259090) &
        (t.dropoff_longitude <= -73.700272) &
        (t.passenger_count < 6)
    ]
    hours = t.trip_time_in_secs.coerce('float64') / 3600.0
    avg_speed_in_mph = t.trip_distance / hours
    d = transform(t, avg_speed_in_mph=avg_speed_in_mph, mycol=avg_speed_in_mph + 1)
    df = compute(d[d.avg_speed_in_mph <= 200], nyc, return_type=pd.DataFrame)
    assert not df.empty


def test_coerce_on_select(nyc):
    t = symbol('t', discover(nyc))
    t = t[
        (t.pickup_latitude >= 40.477399) &
        (t.pickup_latitude <= 40.917577) &
        (t.dropoff_latitude >= 40.477399) &
        (t.dropoff_latitude <= 40.917577) &
        (t.pickup_longitude >= -74.259090) &
        (t.pickup_longitude <= -73.700272) &
        (t.dropoff_longitude >= -74.259090) &
        (t.dropoff_longitude <= -73.700272) &
        (t.passenger_count < 6)
    ]
    t = transform(t, pass_count=t.passenger_count + 1)
    result = compute(t.pass_count.coerce('float64'), nyc, return_type='native')
    s = odo(result, pd.Series)
    expected = compute(t, nyc, return_type=pd.DataFrame) \
                      .passenger_count.astype('float64') + 1.0
    assert list(s) == list(expected)


def test_interactive_len(sql):
    t = Data(sql)
    assert len(t) == int(t.count())


def test_core_compute(nyc):
    t = symbol('t', discover(nyc))
    assert isinstance(compute(t, nyc, return_type='core'), pd.DataFrame)
    assert isinstance(compute(t.passenger_count, nyc, return_type='core'), pd.Series)
    assert iscorescalar(compute(t.passenger_count.mean(), nyc, return_type='core'))
    assert isinstance(compute(t, nyc, return_type=list), list)


def test_sample(sql):
    t = symbol('t', discover(sql))
    result = compute(t.sample(n=1), sql)
    s = odo(result, pd.DataFrame)
    assert len(s) == 1
    result2 = compute(t.sample(frac=0.5), sql)
    s2 = odo(result2, pd.DataFrame)
    assert len(s) == len(s2)
