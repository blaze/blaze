from __future__ import absolute_import, division, print_function

import pytest

pymysql = pytest.importorskip('pymysql')

import sqlalchemy

from blaze import SQL, CSV, drop
from blaze.api.into import into
from blaze.utils import filetext, assert_allclose
from blaze.compatibility import xfail
import os
import pandas as pd
import datetime as dt
import numpy as np
import getpass


username = getpass.getuser()
url = 'mysql+pymysql://{0}@localhost:3306/test'.format(username)


@pytest.yield_fixture
def csv():
    with filetext('1,2\n10,20\n100,200', '.csv') as f:
        yield CSV(f, columns=list('ab'))


@pytest.yield_fixture
def csv_no_columns():
    with filetext('1,2\n10,20\n100,200', '.csv') as f:
        yield CSV(f)


@pytest.yield_fixture
def float_csv():
    with filetext('1.02,2.02\n102.02,202.02\n1002.02,2002.02', '.csv') as f:
        yield CSV(f, columns=list('ab'))


@pytest.fixture
def complex_csv():
    this_dir = os.path.dirname(__file__)
    file_name = os.path.join(this_dir, 'dummydata.csv')
    return CSV(file_name, schema='{Name: string, RegistrationDate: date, ZipCode: int64, Consts: float64}')


@pytest.yield_fixture
def sql(csv):
    name = 'test_table'
    s = SQL(url, name, schema=csv.schema)
    engine = s.engine
    yield s
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine, only=[s.tablename])
    t = metadata.tables[s.tablename]
    t.drop(engine)


@pytest.yield_fixture
def float_sql(float_csv):
    name = 'float_table'
    s = SQL(url, name, schema=float_csv.schema)
    engine = s.engine
    yield s
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine, only=[s.tablename])
    t = metadata.tables[s.tablename]
    t.drop(engine)


@pytest.yield_fixture
def complex_sql(complex_csv):
    name = 'complex_test_table'
    s = SQL(url, name, schema=complex_csv.schema)
    engine = s.engine
    yield s
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine, only=[s.tablename])
    t = metadata.tables[s.tablename]
    t.drop(engine)


def test_csv_postgres_load(sql, csv):
    engine = sql.engine
    conn = engine.raw_connection()
    cursor = conn.cursor()
    full_path = os.path.abspath(csv.path)
    load = (r"LOAD DATA INFILE '{0}' INTO TABLE {1} FIELDS TERMINATED BY ',' "
            "lines terminated by '\n'").format(full_path, sql.tablename)
    cursor.execute(load)
    conn.commit()


def test_simple_into(sql, csv):
    into(sql, csv, if_exists="replace")

    a = list(sql[:, 'a'])
    b = list(sql[:, 'b'])

    assert a == [1, 10, 100]
    assert b == [2, 20, 200]


def test_append(sql, csv):
    into(sql, csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]

    into(sql, csv, if_exists="append")

    assert list(sql[:, 'a']) == [1, 10, 100, 1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200, 2, 20, 200]


def test_simple_float_into(float_sql, float_csv):
    into(float_sql, float_csv, if_exists="replace")

    assert list(float_sql[:, 'a']) == [1.02, 102.02, 1002.02]
    assert list(float_sql[:, 'b']) == [2.02, 202.02, 2002.02]


def test_tryexcept_into(sql, csv):
    # uses multi-byte character and fails over to using sql.extend()
    into(sql, csv, if_exists="replace", QUOTE="alpha", FORMAT="csv")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]


@xfail(raises=KeyError)
def test_failing_argument(sql, csv):
    into(sql, csv, if_exists="replace", skipinitialspace="alpha") # failing call


def test_no_header_no_columns(csv_no_columns):
    sql = SQL(url, 'test_table', schema='{x: int, y: int}')
    into(sql, csv_no_columns, if_exists="replace")

    assert list(sql[:, 'x']) == [1, 10, 100]
    assert list(sql[:, 'y']) == [2, 20, 200]


def test_complex_into(complex_sql, complex_csv):
    # data from: http://dummydata.me/generate

    sql, csv = complex_sql, complex_csv
    into(sql, csv, if_exists="replace")

    df = pd.read_csv(csv.path, parse_dates=['RegistrationDate'])

    assert_allclose([sql[0]], [csv[0]])

    # TODO: implement count method

    for col in sql.columns:
        # need to convert to python datetime
        if col == "RegistrationDate":
            py_dates = list(df['RegistrationDate'].astype(object).values)
            py_dates = [dt.date(d.year, d.month, d.day) for d in py_dates]
            assert list(sql[:, col]) == list(csv[:, col]) == py_dates

        # handle floating point precision -- perhaps it's better to call out to
        # assert_array_almost_equal
        elif col == 'Consts':

            # WARNING!!! Floats are truncated with MySQL and the assertion fails
            sql_array = np.array(list(sql[:, col]))
            csv_array = list(csv[:, col])
            df_array = df[col].values
            np.testing.assert_almost_equal(sql_array, csv_array, decimal=5)
            np.testing.assert_almost_equal(sql_array, df_array, decimal=5)
        else:
            assert list(sql[:, col]) == list(csv[:,
                                                 col]) == list(df[col].values)
