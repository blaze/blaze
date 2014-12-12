from __future__ import absolute_import, division, print_function

import pytest


from datashape import dshape
from blaze import CSV
from into import into, resource
from blaze.utils import tmpfile
import sqlalchemy


@pytest.yield_fixture
def engine():
    tbl = 'testtable'


ds = dshape('var *  {a: int32, b: int32}' )
data = [(1, 2), (10, 20), (100, 200)]

@pytest.yield_fixture
def csv():
    with tmpfile('csv') as filename:
        csv = into(filename, data, dshape=ds)
        yield csv


def test_simple_into(csv):
    tbl = 'testtable'
    with tmpfile('db') as filename:
        engine = sqlalchemy.create_engine('sqlite:///' + filename)
        t = resource('sqlite:///' + filename + '::' + tbl,
                     dshape=ds)

        into(t, csv, dshape=ds)
        conn = engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name='{0}';".format(tbl))

        sqlite_tbl_names = cursor.fetchall()
        assert sqlite_tbl_names[0][0] == tbl

        assert into(list, t) == data
