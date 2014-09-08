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

file_name = 'test.csv'

@pytest.yield_fixture
def engine():
    with tmpfile('db') as filename:
        engine = sqlalchemy.create_engine('sqlite:///' + filename)
        yield engine


def setup_function(function):
    data = [(1, 2), (10, 20), (100, 200)]

    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)


def teardown_function(function):
    os.remove(file_name)


def test_csv_sqlite_load(engine):

    tbl = 'testtable'

    csv = CSV(file_name)

    # how to handle path to DB.
    sql = SQL(engine, tbl, schema=csv.schema)
    engine = sql.engine
    conn = engine.raw_connection()

    dbtype = sql.engine.url.drivername
    db = sql.engine.url.database
    engine = sql.engine
    abspath = csv._abspath
    tblname = sql.tablename

    copy_info = {'abspath': abspath,
                 'tblname': tblname,
                 'db': db,
                }

    copy_cmd = "(echo '.mode csv'; echo '.import {abspath} {tblname}';) | sqlite3 {db}"
    copy_cmd = copy_cmd.format(**copy_info)

    ps = subprocess.Popen(copy_cmd,shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    assert list(csv) == list(sql)


def test_simple_into(engine):

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(engine, tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")
    conn = sql.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name='{0}';".format(tbl))

    sqlite_tbl_names = cursor.fetchall()
    assert sqlite_tbl_names[0][0] == tbl


    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]
