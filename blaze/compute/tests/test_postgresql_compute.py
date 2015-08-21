from datetime import timedelta
from operator import methodcaller
import itertools
import re

import pytest

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('psycopg2')

import numpy as np
import pandas as pd

import pandas.util.testing as tm

from datashape import dshape
from odo import odo, resource, drop, discover
from blaze import symbol, compute, concat


names = ('tbl%d' % i for i in itertools.count())


def normalize(s):
    s = ' '.join(s.strip().split()).lower()
    s = re.sub(r'(alias)_?\d*', r'\1', s)
    return re.sub(r'__([A-Za-z_][A-Za-z_0-9]*)', r'\1', s)


@pytest.fixture
def name():
    return next(names)


@pytest.fixture(scope='session')
def base_url():
    return 'postgresql://postgres@localhost/test::%s'


@pytest.fixture
def url(base_url, name):
    return base_url % name


@pytest.yield_fixture
def sql(url):
    try:
        t = resource(url, dshape='var * {A: string, B: int64}')
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
        t = resource(url, dshape='var * {A: datetime}')
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        t = odo([(d,) for d in pd.date_range('2014-01-01', '2014-02-01')], t)
        try:
            yield t
        finally:
            drop(t)


@pytest.yield_fixture
def sql_two_tables(base_url):
    dshape = 'var * {a: int32}'
    try:
        t = resource(url(base_url, next(names)), dshape=dshape)
        u = resource(url(base_url, next(names)), dshape=dshape)
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield u, t
        finally:
            drop(t)
            drop(u)


@pytest.yield_fixture
def products(base_url):
    try:
        products = resource(base_url % 'products',
                            dshape="""var * {
                                product_id: !int64,
                                color: ?string,
                                price: float64
                            }""")
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield products
        finally:
            drop(products)


@pytest.yield_fixture
def orders(base_url, products):
    try:
        orders = resource(base_url % 'orders',
                          dshape="""var * {
                            order_id: !int64,
                            product_id: map[int64, T],
                            quantity: int64
                          }
                          """, foreign_keys=dict(product_id=products.c.product_id))
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield orders
        finally:
            drop(orders)


@pytest.yield_fixture
def main(base_url):
    try:
        main = odo([(i, np.random.randint(10)) for i in range(13)],
                   base_url % 'main',
                   dshape=dshape('var * {id: !int64, data: int64}'))
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield main
        finally:
            drop(main)


@pytest.yield_fixture
def pkey(base_url, main):
    choices = [u'AAPL', u'HPQ', u'ORCL', u'IBM', u'DOW', u'SBUX', u'AMD',
               u'INTC', u'GOOG', u'PRU', u'MSFT', u'AIG', u'TXN', u'DELL',
               u'PEP']
    n = 100
    data = list(zip(np.arange(n),
                    np.random.choice(choices, size=n),
                    np.random.uniform(10000, 20000, size=n),
                    np.random.randint(main.count().scalar(), size=n)))
    try:
        main = odo(data, base_url % 'pkey',
                   dshape=dshape('var * {id: !int64, sym: string, price: float64, main: map[int64, T]}'),
                   foreign_keys=dict(main=main.c.id))
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield main
        finally:
            drop(main)


@pytest.yield_fixture
def fkey(base_url, pkey):
    try:
        main = odo([(i,
                     np.random.randint(pkey.count().scalar()),
                     np.random.randint(10000))
                    for i in range(10)],
                   base_url % 'fkey',
                   dshape=dshape('var * {id: !int64, sym_id: map[int64, T], size: int64}'),
                   foreign_keys=dict(sym_id=pkey.c.id))
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield main
        finally:
            drop(main)


@pytest.yield_fixture
def sql_with_float(url):
    try:
        t = resource(url, dshape='var * {c: float64}')
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


def test_auto_join_field(orders):
    t = symbol('t', discover(orders))
    expr = t.product_id.color
    result = compute(expr, orders)
    expected = """SELECT
        products.color
    FROM products, orders
    WHERE orders.product_id = products.product_id
    """
    assert normalize(str(result)) == normalize(expected)


def test_auto_join_projection(orders):
    t = symbol('t', discover(orders))
    expr = t.product_id[['color', 'price']]
    result = compute(expr, orders)
    expected = """SELECT
        products.color,
        products.price
    FROM products, orders
    WHERE orders.product_id = products.product_id
    """
    assert normalize(str(result)) == normalize(expected)


@pytest.mark.xfail(raises=NotImplementedError,
                   reason='Not yet implemented')
@pytest.mark.parametrize('func', ['max', 'min', 'sum'])
def test_foreign_key_reduction(orders, products, func):
    t = symbol('t', discover(orders))
    expr = methodcaller(func)(t.quantity * t.product_id.price)
    result = compute(expr, orders)
    expected = """SELECT
        max(orders.quantity * products.price) AS max
    FROM orders, products WHERE orders.product_id = products.product_id
    """.format(func)
    assert normalize(str(result)) == normalize(expected)


def test_foreign_key_chain(fkey):
    t = symbol('t', discover(fkey))
    expr = t.sym_id.main.data
    result = compute(expr, fkey)
    expected = """SELECT
        main.data
    FROM main, fkey, pkey
    WHERE fkey.sym_id = pkey.id and pkey.main = main.id
    """
    assert normalize(str(result)) == normalize(expected)


def test_foreign_key_group_by(fkey):
    t = symbol('fkey', discover(fkey))
    expr = by(t.sym_id.sym, avg_price=t.sym_id.price.mean())
    result = compute(expr, fkey)
    expected = """SELECT
        pkey.sym,
        avg(pkey.price) AS avg_price
    FROM pkey, fkey
    WHERE fkey.sym_id = pkey.id
    GROUP BY pkey.sym
    """
    assert normalize(str(result)) == normalize(expected)
