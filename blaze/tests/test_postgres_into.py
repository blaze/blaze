from __future__ import absolute_import, division, print_function

import pytest

psycopg2 = pytest.importorskip('psycopg2')
import subprocess
ps = subprocess.Popen("ps aux | grep postgres",shell=True, stdout=subprocess.PIPE)
output = ps.stdout.read()
num_processes = len(output.splitlines())
pytestmark = pytest.mark.skipif(num_processes < 6, reason="No Postgres Installation")


from blaze import SQL
from blaze import CSV
from blaze.api.into import into
from blaze.utils import assert_allclose
import sqlalchemy
import os
import csv as csv_module
import pandas as pd
import datetime as dt
import numpy as np

url = 'postgresql://localhost/postgres'
file_name = 'test.csv'

def setup_function(function):
    data = [(1, 2), (10, 20), (100, 200)]

    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)

def teardown_function(function):
    os.remove(file_name)
    engine = sqlalchemy.create_engine(url)
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine)

    for t in metadata.tables:
        if 'testtable' in t:
            metadata.tables[t].drop(engine)

def test_csv_postgres_load():

    tbl = 'testtable'

    engine = sqlalchemy.create_engine(url)

    if engine.has_table(tbl):
        metadata = sqlalchemy.MetaData()
        metadata.reflect(engine)
        t = metadata.tables[tbl]
        t.drop(engine)

    csv = CSV(file_name)

    sql = SQL(url,tbl, schema=csv.schema)
    engine = sql.engine
    conn = engine.raw_connection()

    cursor = conn.cursor()
    full_path = os.path.abspath(file_name)
    load = '''copy {0} from '{1}'(FORMAT CSV, DELIMITER ',', NULL '');'''.format(tbl, full_path)
    cursor.execute(load)
    conn.commit()


def test_simple_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]

def test_append():

    tbl = 'testtable_into_append'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]

    into(sql,csv, if_exists="append")
    assert list(sql[:, 'a']) == [1, 10, 100, 1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200, 2, 20, 200]


def test_tryexcept_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace", QUOTE="alpha", FORMAT="csv") # uses multi-byte character and
                                                      # fails over to using sql.extend()

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]


@pytest.mark.xfail(raises=KeyError)
def test_failing_argument():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace", skipinitialspace="alpha") # failing call

def test_no_header_no_columns():

    tbl = 'testtable_into_2'

    csv = CSV(file_name)
    sql = SQL(url,tbl, schema= '{x: int, y: int}')
    # import pdb
    # pdb.set_trace()
    into(sql,csv, if_exists="replace")

    assert list(sql[:, 'x']) == [1, 10, 100]
    assert list(sql[:, 'y']) == [2, 20, 200]

def test_complex_into():
    # data from: http://dummydata.me/generate

    this_dir = os.path.dirname(__file__)
    file_name = os.path.join(this_dir, 'dummydata.csv')

    tbl = 'testtable_into_complex'

    csv = CSV(file_name, schema='{Name: string, RegistrationDate: date, ZipCode: int32, Consts: float64}')
    sql = SQL(url, tbl, schema=csv.schema)

    into(sql, csv, if_exists="replace")

    df = pd.read_csv(file_name, parse_dates=['RegistrationDate'])

    assert_allclose([sql[0]], [csv[0]])

    for col in sql.columns:
        # need to convert to python datetime
        if col == "RegistrationDate":
            py_dates = list(df['RegistrationDate'].map(lambda x: x.date()).values)
            assert list(sql[:, col]) == list(csv[:, col]) == py_dates
        elif col == 'Consts':
            l, r = list(sql[:, col]), list(csv[:, col])
            assert np.allclose(l, df[col].values)
            assert np.allclose(l, r)
        else:
            assert list(sql[:, col]) == list(csv[:,col]) == list(df[col].values)
