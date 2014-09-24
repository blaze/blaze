from __future__ import absolute_import, division, print_function

import pytest
from dynd import nd
import numpy as np
import tables as tb
from pandas import DataFrame

from blaze.api.into import into
from blaze.api.into import degrade_numpy_dtype_to_python, numpy_ensure_bytes
from blaze.utils import tmpfile
from blaze import Table, resource
import bcolz
from blaze.data import CSV
from blaze.sql import SQL
from datetime import datetime
from toolz import pluck
import os

dirname = os.path.dirname(__file__)


csv = CSV(os.path.join(dirname, 'accounts.csv'))

L = [[100, 1, 'Alice', datetime(2000, 12, 25, 0, 0, 1)],
     [200, 2, 'Bob', datetime(2001, 12, 25, 0, 0, 1)],
     [300, 3, 'Charlie', datetime(2002, 12, 25, 0, 0, 1)]]

df = DataFrame(L, columns=['amount', 'id', 'name', 'timestamp'])

x = np.array(list(map(tuple, L)),
             dtype=[('amount', 'i8'), ('id', 'i8'),
                    ('name', 'U7'), ('timestamp', 'M8[us]')])

schema = '{amount: int64, id: int64, name: string, timestamp: datetime}'
sql_schema = '{amount: int64, id: int64, name: string, timestamp: datetime[tz="UTC"]}'

arr = nd.array(L, dtype=schema)

bc = bcolz.ctable([np.array([100, 200, 300], dtype=np.int64),
                   np.array([1, 2, 3], dtype=np.int64),
                   np.array(['Alice', 'Bob', 'Charlie'], dtype='U7'),
                   np.array([datetime(2000, 12, 25, 0, 0, 1),
                             datetime(2001, 12, 25, 0, 0, 1),
                             datetime(2002, 12, 25, 0, 0, 1)], dtype='M8[us]')],
                  names=['amount', 'id', 'name', 'timestamp'])

sql = SQL('sqlite:///:memory:', 'accounts', schema=schema)
sql.extend(L)

data = [(list, L),
        (Table, Table(L, '{amount: int64, id: int64, name: string[7], timestamp: datetime}')),
        (DataFrame, df),
        (np.ndarray, x),
        (nd.array, arr),
        (bcolz.ctable, bc),
        (CSV, csv),
        (SQL, sql)]

schema_no_date = '{amount: int64, id: int64, name: string[7]}'
sql_no_date = SQL('sqlite:///:memory:', 'accounts_no_date', schema=schema_no_date)

L_no_date = list(pluck([0, 1, 2], L))
sql_no_date.extend(L_no_date)

no_date = [(list, list(pluck([0, 1, 2], L))),
           (Table, Table(list(pluck([0, 1, 2], L)),
                         '{amount: int64, id: int64, name: string[7]}')),
           (DataFrame, df[['amount', 'id', 'name']]),
           (np.ndarray, x[['amount', 'id', 'name']]),
           (nd.array, nd.fields(arr, 'amount', 'id', 'name')),
           (bcolz.ctable, bc[['amount', 'id', 'name']]),
           (SQL, sql_no_date)]


try:
    import pymongo
except ImportError:
    pymongo = None
    Collection = None
if pymongo:
    from pymongo.collection import Collection
    try:
        db = pymongo.MongoClient().db

        db.test.drop()
        data.append((Collection, into(db.test, df)))

        db.no_date.drop()
        no_date.append((Collection, into(db.no_date, dict(no_date)[DataFrame])))
    except pymongo.errors.ConnectionFailure:
        pymongo = None
        Collection = None

try:
    import tables
    f = tables.open_file(os.path.join(dirname, 'accounts.h5'))
    pytab = tb = f.get_node('/accounts')
    no_date.append((tables.Table, tb))
    from tables import Table as PyTable
except ImportError:
    tables = None
    PyTable = None


def normalize(a):
    """ Normalize results prior to equality test

    Ensure that (1, 2, 3) == [1, 2, 3] and that u'Hello' == 'Hello'
    """
    if isinstance(a, np.ndarray):
        a = a.astype(degrade_numpy_dtype_to_python(a.dtype))
    if isinstance(a, bcolz.ctable):
        return normalize(a[:])
    if isinstance(a, SQL):
        return list(a)
    return (str(a).replace("u'", "'")
                  .replace("(", "[").replace(")", "]")
                  .replace('L', ''))


def test_base():
    """ Test all pairs of base in-memory data structures """
    sources = [v for k, v in data if k not in [list]]
    targets = [v for k, v in data if k not in [Table, Collection, CSV,
        nd.array, SQL]]
    for a in sources:
        for b in targets:
            assert normalize(into(type(b), a)) == normalize(b)

def test_into_empty_sql():
    """ Test all sources into empty SQL database """
    sources = [v for k, v in data if k not in [list]]
    for a in sources:
        sql_empty = resource('sqlite:///:memory:', 'accounts',
                             schema=sql_schema)
        assert normalize(into(sql_empty, a)) == normalize(sql)


def test_expressions():
    sources = [v for k, v in data if k not in [nd.array, CSV, Table]]
    targets = [v for k, v in no_date if k not in
               [Table, CSV, Collection, nd.array, PyTable, SQL]]

    for a in sources:
        for b in targets:
            c = Table(a, "{amount: int64, id: int64, name: string, timestamp: datetime}")[['amount', 'id', 'name']]
            assert normalize(into(type(b), c)) == normalize(b)

try:
    from bokeh.objects import ColumnDataSource
    cds = ColumnDataSource({
         'id': [1, 2, 3],
         'name': ['Alice', 'Bob', 'Charlie'],
         'amount': [100, 200, 300],
         'timestamp': [datetime(2000, 12, 25, 0, 0, 1),
                       datetime(2001, 12, 25, 0, 0, 1),
                       datetime(2002, 12, 25, 0, 0, 1)]
         })
except ImportError:
    ColumnDataSource = None


def skip_if_not(x):
    """ Possibly skip test if type is not present """
    def maybe_a_test_function(test_foo):
        if not x:
            return
        else:
            return test_foo
    return maybe_a_test_function


@skip_if_not(ColumnDataSource)
def test_ColumnDataSource():
    sources = [v for k, v in data if k not in [list]]
    for a in sources:
        assert into(ColumnDataSource, a).data == cds.data


@pytest.yield_fixture
def h5tmp():
    with tmpfile('.h5') as filename:
        yield filename


tables_data = [v for k, v in data if k != list]


@pytest.mark.parametrize('a', tables_data)
def test_into_PyTables(a, h5tmp):
    dshape = 'var * {amount: int64, id: int64, name: string[7, "A"], timestamp: datetime}'
    lhs = into(tables.Table, a, dshape=dshape, filename=h5tmp, datapath='/data')
    np.testing.assert_array_equal(into(np.ndarray, lhs), numpy_ensure_bytes(x))
    lhs._v_file.close()


@pytest.fixture
def mconn():
    pymongo = pytest.importorskip('pymongo')
    try:
        c = pymongo.MongoClient()
    except pymongo.errors.ConnectionFailure:
        pytest.skip('unable to connect')
    else:
        return c


@pytest.fixture
def mdb(mconn):
    return mconn.db


def test_mongo_Collection(mdb):
    sources = [v for k, v in data if k not in [list]]
    for a in sources:
        mdb.test_into.drop()
        into(mdb.test_into, a)
        assert normalize(into([], mdb.test_into)) == normalize(L)
