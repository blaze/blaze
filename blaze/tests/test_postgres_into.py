from __future__ import absolute_import, division, print_function

import pytest

psycopg2 = pytest.importorskip('psycopg2')


from blaze import SQL, CSV, into, resource
from blaze.api.into import into
from blaze.utils import assert_allclose, filetext, chmod, WORLD_READABLE
import os
import pandas as pd
import numpy as np


url = 'postgresql://localhost/postgres'


@pytest.yield_fixture
def csv_no_header():
    with filetext('1,2\n10,20\n100,200', '.csv') as f:
        with chmod(f, flags=WORLD_READABLE) as g:
            yield CSV(g, columns=list('ab'))


@pytest.yield_fixture
def sql_no_header():
    yield resource(url, 'test_table', schema='{x: int, y: int}')


def test_csv_postgres_load(sql, csv):
    full_path = os.path.abspath(csv.path)
    load = '''copy {0} from '{1}'(FORMAT CSV, DELIMITER ',', NULL '');'''.format(sql.tablename, full_path)
    with sql.engine.begin() as conn:
        conn.execute(load)


def test_simple_into(sql, csv):
    into(sql, csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]


def test_append(sql, csv):
    into(sql, csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]

    into(sql, csv, if_exists="append")
    assert list(sql[:, 'a']) == [1, 10, 100, 1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200, 2, 20, 200]


def test_tryexcept_into(sql, csv):
    # should use extend here
    into(sql, csv, if_exists="replace", QUOTE="alpha", FORMAT="csv")
    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]


@pytest.mark.xfail(raises=KeyError)
def test_failing_argument(sql, csv):
    into(sql, csv, if_exists="replace", skipinitialspace="alpha") # failing call


def test_no_header_no_columns(sql_no_header, csv_no_header):
    into(sql_no_header, csv_no_header, if_exists="replace")

    assert list(sql_no_header[:, 'x']) == [1, 10, 100]
    assert list(sql_no_header[:, 'y']) == [2, 20, 200]


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
