from __future__ import absolute_import, division, print_function

import pytest


from blaze import SQL
from blaze import CSV
from blaze.api.into import into
from blaze.utils import tmpfile
import sqlalchemy
import os
import csv as csv_module
import subprocess


@pytest.yield_fixture
def engine():
    with tmpfile('db') as filename:
        engine = sqlalchemy.create_engine('sqlite:///' + filename)
        yield engine


@pytest.yield_fixture
def csv():
    data = [(1, 2), (10, 20), (100, 200)]

    with tmpfile('csv') as filename:
        csv = CSV(filename, 'w', schema='{a: int32, b: int32}')
        csv.extend(data)
        csv = CSV(filename, schema='{a: int32, b: int32}')
        yield csv


def test_simple_into(engine, csv):

    tbl = 'testtable_into_2'

    sql = SQL(engine, tbl, schema= csv.schema)

    into(sql, csv, if_exists="replace")
    conn = sql.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name='{0}';".format(tbl))

    sqlite_tbl_names = cursor.fetchall()
    assert sqlite_tbl_names[0][0] == tbl

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]
