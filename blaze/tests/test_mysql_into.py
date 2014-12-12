from __future__ import absolute_import, division, print_function

import pytest

pymysql = pytest.importorskip('pymysql')
import subprocess
ps = subprocess.Popen("ps aux | grep '[m]ysql'",shell=True, stdout=subprocess.PIPE)
output = ps.stdout.read()
num_processes = len(output.splitlines())
pytestmark = pytest.mark.skipif(num_processes < 3, reason="No MySQL Installation")


from blaze import CSV, resource, into
import sqlalchemy
import os
import csv as csv_module
import pandas as pd
import getpass

username = getpass.getuser()
url = 'mysql+pymysql://{0}@localhost:3306/test'.format(username)
file_name = 'test.csv'
file_name_floats = 'test_floats.csv'

def create_csv(data, file_name):
    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)


def setup_function(function):
    data = [(1, 2), (10, 20), (100, 200)]
    data_floats = [(1.02, 2.02), (102.02, 202.02), (1002.02, 2002.02)]

    create_csv(data,file_name)
    create_csv(data_floats,file_name_floats)


def teardown_function(function):
    os.remove(file_name)
    os.remove(file_name_floats)
    engine = sqlalchemy.create_engine(url)
    metadata = sqlalchemy.MetaData()
    metadata.reflect(engine)

    for t in metadata.tables:
        if 'testtable' in t:
            # pass
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

    sql = resource(url + '::' + tbl, dshape=csv.dshape)
    engine = sql.bind
    conn = engine.raw_connection()

    cursor = conn.cursor()
    full_path = os.path.abspath(file_name)
    load = '''LOAD DATA INFILE '{0}' INTO TABLE {1} FIELDS TERMINATED BY ','
        lines terminated by '\n'
        '''.format(full_path, tbl)
    cursor.execute(load)
    conn.commit()


def test_simple_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql, csv, if_exists="replace")

    assert into(list, sql) == [(1, 2), (10, 20), (100, 200)]


def test_append():

    tbl = 'testtable_into_append'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql, csv, if_exists="replace")
    assert into(list, sql) == [(1, 2), (10, 20), (100, 200)]

    into(sql, csv, if_exists="append")
    assert into(list, sql) == [(1, 2), (10, 20), (100, 200),
                               (1, 2), (10, 20), (100, 200)]


def test_simple_float_into():
    tbl = 'testtable_into_float'

    csv = CSV(file_name_floats, columns=['a', 'b'])
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql,csv, if_exists="replace")

    assert into(list, sql) == \
            [(1.02, 2.02), (102.02, 202.02), (1002.02, 2002.02)]

def test_tryexcept_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql, csv, if_exists="replace", QUOTE="alpha", FORMAT="csv") # uses multi-byte character and
                                                      # fails over to using sql.extend()

    assert into(list, sql) == [(1, 2), (10, 20), (100, 200)]


@pytest.mark.xfail(raises=KeyError)
def test_failing_argument():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql, csv, if_exists="replace", skipinitialspace="alpha") # failing call


def test_no_header_no_columns():
    tbl = 'testtable_into_2'

    csv = CSV(file_name)
    sql = resource(url + '::' + tbl, dshape=csv.dshape)

    into(sql, csv, if_exists="replace")

    assert into(list, sql) == [(1, 2), (10, 20), (100, 200)]


def test_complex_into():
    # data from: http://dummydata.me/generate

    this_dir = os.path.dirname(__file__)
    file_name = os.path.join(this_dir, 'dummydata.csv')

    tbl = 'testtable_into_complex'

    csv = CSV(file_name, schema='{Name: string, RegistrationDate: date, ZipCode: int64, Consts: float64}')

    sql = resource(url + '::' + tbl, dshape=csv.dshape)
    into(sql, csv, if_exists="replace")

    df = pd.read_csv(file_name, parse_dates=['RegistrationDate'])

    assert into(list, sql) == into(list, csv)
