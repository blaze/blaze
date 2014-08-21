from __future__ import absolute_import, division, print_function

from dynd import nd
import numpy as np
from pandas import DataFrame

from blaze.api.into import into, discover
from blaze.api.into import degrade_numpy_dtype_to_python
from datashape import dshape
from blaze.bcolz import *
import blaze
from blaze import Table, TableExpr, TableSymbol, compute
import bcolz
from blaze.data import CSV
from datetime import datetime
from toolz import pluck


csv = CSV('blaze/api/tests/accounts.csv')

L = [[100, 1, 'Alice', datetime(2000, 12, 25, 0, 0, 1)],
     [200, 2, 'Bob', datetime(2001, 12, 25, 0, 0, 1)],
     [300, 3, 'Charlie', datetime(2002, 12, 25, 0, 0, 1)]]

df = DataFrame(L, columns=['amount', 'id', 'name', 'timestamp'])

x = np.array(list(map(tuple, L)),
             dtype=[('amount', 'i8'), ('id', 'i8'),
                    ('name', 'U7'), ('timestamp', 'M8[us]')])

schema = '{amount: int64, id: int64, name: string, timestamp: datetime}'

arr = nd.array(L, dtype=schema)

bc = bcolz.ctable([np.array([100, 200, 300], dtype=np.int64),
                   np.array([1, 2, 3], dtype=np.int64),
                   np.array(['Alice', 'Bob', 'Charlie'], dtype='U7'),
                   np.array([datetime(2000, 12, 25, 0, 0, 1),
                             datetime(2001, 12, 25, 0, 0, 1),
                             datetime(2002, 12, 25, 0, 0, 1)], dtype='M8[us]')],
                  names=['amount', 'id', 'name', 'timestamp'])

data = {list: L,
        Table: Table(L, '{amount: int64, id: int64, name: string[7], timestamp: datetime}'),
        DataFrame: df,
        np.ndarray: x,
        nd.array: arr,
        bcolz.ctable: bc,
        CSV: csv}

no_date = {list: list(pluck([0, 1, 2], L)),
           Table: Table(list(pluck([0, 1, 2], L)),
                        '{amount: int64, id: int64, name: string[7]}'),
           DataFrame: df[['amount', 'id', 'name']],
           np.ndarray: x[['amount', 'id', 'name']],
           nd.array: nd.fields(arr, 'amount', 'id', 'name'),
           bcolz.ctable: bc[['amount', 'id', 'name']]}


try:
    import pymongo
    from pymongo import Collection
    db = pymongo.MongoClient().db

    db.test.drop()
    data[pymongo.Collection] = into(db.test, df)

    db.no_date.drop()
    no_date[pymongo.Collection] = into(db.no_date, no_date[DataFrame])
except ImportError:
    pymongo = None
    Collection = None


def normalize(a):
    """ Normalize results prior to equality test

    Ensure that (1, 2, 3) == [1, 2, 3] and that u'Hello' == 'Hello'
    """
    if isinstance(a, np.ndarray):
        a = a.astype(degrade_numpy_dtype_to_python(a.dtype))
    if isinstance(a, bcolz.ctable):
        return normalize(a[:])
    return (str(a).replace("u'", "'")
                  .replace("(", "[").replace(")", "]")
                  .replace('L', ''))


def test_base():
    """ Test all pairs of base in-memory data structures """
    sources = [v for k, v in data.items() if k not in [list]]
    targets = [v for k, v in data.items() if k not in [Table, Collection]]
    for a in sources:
        for b in targets:
            assert normalize(into(type(b), a)) == normalize(b)


def test_expressions():
    sources = [v for k, v in data.items() if k not in [nd.array, CSV, Table]]
    targets = [v for k, v in no_date.items() if k not in [Table, CSV,
        Collection, nd.array]]

    for a in sources:
        for b in targets:
            c = Table(a, '{amount: int64, id: int64, name: string[7], timestamp: datetime}')[['amount', 'id', 'name']]
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
    sources = [v for k, v in data.items() if k not in [list]]
    for a in sources:
        assert into(ColumnDataSource, a).data == cds.data


@skip_if_not(pymongo)
def test_mongo_Collection():
    sources = [v for k, v in data.items() if k not in [list]]
    for a in sources:
        db.test_into.drop()
        into(db.test_into, a)
        assert normalize(into([], db.test_into)) == normalize(L)
