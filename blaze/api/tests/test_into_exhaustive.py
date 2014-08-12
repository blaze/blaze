from __future__ import absolute_import, division, print_function

from dynd import nd
import numpy as np
from pandas import DataFrame

from blaze.api.into import into, discover
from datashape import dshape
from blaze.bcolz import *
import blaze
from blaze import Table
import bcolz


L = [[100, 1, 'Alice'],
     [200, 2, 'Bob'],
     [300, 3, 'Charlie']]

df = DataFrame(L, columns=['amount', 'id', 'name'])

x = np.array(list(map(tuple, L)), dtype=[('amount', 'i8'), ('id', 'i8'), ('name', 'U7')])

arr = nd.array(L, dtype='{amount: int64, id: int64, name: string}')

bc = bcolz.ctable([np.array([100, 200, 300], dtype=np.int64),
                   np.array([1, 2, 3], dtype=np.int64),
                   np.array(['Alice', 'Bob', 'Charlie'], dtype='U7')],
                  names=['amount', 'id', 'name'])

sources = [Table(L, schema='{amount: int64, id: int64, name: string[7]}'),
             df, x, arr, bc]
targets = [L, df, x, arr, bc]

try:
    import pymongo
    db = pymongo.MongoClient().db
    db.test.drop()
    into(db.test, df)
    sources.append(db.test)
except ImportError:
    pymongo = None


def normalize(a):
    return str(a).replace("u'", "'").replace("(", "[").replace(")", "]")


def test_base():
    for a in sources:
        for b in targets:
            assert normalize(into(type(b), a)) == normalize(b)

try:
    from bokeh.objects import ColumnDataSource
    cds = ColumnDataSource({
         'id': [1, 2, 3],
         'name': ['Alice', 'Bob', 'Charlie'],
         'amount': [100, 200, 300]
         })
except ImportError:
    ColumnDataSource = None


def skip_if_not(x):
    def maybe_a_test_function(test_foo):
        if not x:
            return
        else:
            return test_foo
    return maybe_a_test_function


@skip_if_not(ColumnDataSource)
def test_ColumnDataSource():
    for a in sources:
        assert into(ColumnDataSource, a).data == cds.data


@skip_if_not(pymongo)
def test_mongo_Collection():
    for a in sources:
        db.test_into.drop()
        into(db.test_into, a)
        assert normalize(into([], db.test_into)) == normalize(L)
