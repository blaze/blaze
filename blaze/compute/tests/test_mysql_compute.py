from __future__ import absolute_import, print_function, division

from getpass import getuser

import pytest

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('pymysql')

from odo import odo, drop, discover

import pandas as pd

from blaze import symbol, compute
from blaze.utils import example, normalize


@pytest.yield_fixture(scope='module')
def data():
    try:
        t = odo(
            example('nyc.csv'),
            'mysql+pymysql://%s@localhost/test::nyc' % getuser()
        )
    except sa.exc.OperationalError as e:
        pytest.skip(str(e))
    else:
        try:
            yield t.bind
        finally:
            drop(t)


@pytest.fixture
def db(data):
    return symbol('test', discover(data))


def test_agg_sql(db, data):
    subset = db.nyc[['pickup_datetime', 'dropoff_datetime', 'passenger_count']]
    expr = subset[subset.passenger_count < 4].passenger_count.min()
    result = compute(expr, data)
    expected = """
    select
        min(alias.passenger_count) as passenger_count_min
    from
        (select
            nyc.passenger_count as passenger_count
         from
            nyc
         where nyc.passenger_count < %s) as alias
    """
    assert normalize(str(result)) == normalize(expected)


def test_agg_compute(db, data):
    subset = db.nyc[['pickup_datetime', 'dropoff_datetime', 'passenger_count']]
    expr = subset[subset.passenger_count < 4].passenger_count.min()
    result = compute(expr, data)
    passenger_count = odo(compute(db.nyc.passenger_count, {db: data}), pd.Series)
    assert passenger_count[passenger_count < 4].min() == result.scalar()
