from __future__ import absolute_import, print_function, division

from getpass import getuser

import pytest

sa = pytest.importorskip('sqlalchemy')
pytest.importorskip('pymysql')

from odo import odo, drop, discover

import pandas as pd
import numpy as np

from blaze import symbol, compute
from blaze.utils import example, normalize
from blaze.interactive import iscoretype, iscorescalar, iscoresequence


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
    result = compute(expr, data, return_type='native')
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
    result = compute(expr, data, return_type='native')
    passenger_count = odo(compute(db.nyc.passenger_count, {db: data}, return_type='native'), pd.Series)
    assert passenger_count[passenger_count < 4].min() == result.scalar()


def test_core_compute(db, data):
    assert isinstance(compute(db.nyc, data, return_type='core'), pd.DataFrame)
    assert isinstance(compute(db.nyc.passenger_count, data, return_type='core'), pd.Series)
    assert iscorescalar(compute(db.nyc.passenger_count.mean(), data, return_type='core'))
    assert isinstance(compute(db.nyc, data, return_type=list), list)
