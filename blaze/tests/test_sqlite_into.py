from __future__ import absolute_import, division, print_function

import pytest


from blaze import CSV, resource
from into import into
from blaze.utils import tmpfile
import sqlalchemy
import os
import csv as csv_module
import subprocess


@pytest.yield_fixture
def engine():
    tbl = 'testtable'
    with tmpfile('db') as filename:
        engine = sqlalchemy.create_engine('sqlite:///' + filename)
        t = resource('sqlite:///' + filename + '::' + tbl,
                     dshape='var * {a: int32, b: int32}')
        yield engine, t


@pytest.yield_fixture
def csv():
    data = [(1, 2), (10, 20), (100, 200)]

    with tmpfile('csv') as filename:
        csv = CSV(filename, 'w', schema='{a: int32, b: int32}')
        csv.extend(data)
        csv = CSV(filename, schema='{a: int32, b: int32}')
        yield csv


def test_simple_into(csv):
    tbl = 'testtable'

    sql = resource('sqlite:///:memory:', tbl, dshape=csv.dshape)
    engine = sql.bind

    into(sql, csv, if_exists="replace")
    conn = engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name='{0}';".format(tbl))

    sqlite_tbl_names = cursor.fetchall()
    assert sqlite_tbl_names[0][0] == tbl


    assert into(list, sql) == [(1, 2), (10, 20), (100, 200)]
