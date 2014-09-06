from __future__ import absolute_import, division, print_function

import pytest

from blaze import SQL
from blaze import CSV
from blaze.api.into import into
import sqlalchemy
import os
import csv as csv_module
import subprocess
from blaze.compatibility import PY3

url = 'sqlite:///:memory:'
file_name = 'test.csv'

# @pytest.fixture(scope='module')
def setup_function(function):
    data = [(1, 2), (10, 20), (100, 200)]

    kwargs = {'newline': ''} if PY3 else {}

    with open(file_name, 'w', **kwargs) as f:
        csv_writer = csv_module.writer(f)
        csv_writer.writerows(data)

def teardown_function(function):
    os.remove(file_name)


@pytest.mark.xfail(os.name == 'nt', reason='no sqlite3 command in windows')
def test_csv_sqlite_load():

    tbl = 'testtable'

    engine = sqlalchemy.create_engine(url)

    if engine.has_table(tbl):
        metadata = sqlalchemy.MetaData()
        metadata.reflect(engine)
        t = metadata.tables[tbl]
        t.drop(engine)

    csv = CSV(file_name)

    # how to handle path to DB.
    sql = SQL(url,tbl, schema=csv.schema)
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
    print(sql[0])
    print(list(sql[:]))


def test_simple_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")
    conn = sql.engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' and name='{0}';".format(tbl))

    sqlite_tbl_names = cursor.fetchall()
    assert sqlite_tbl_names[0][0] == tbl


    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]

