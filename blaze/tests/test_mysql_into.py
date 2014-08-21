from __future__ import absolute_import, division, print_function

# A "/etc/my.cnf" from another install may interfere with a Homebrew-built
# server starting up correctly.
#
# To connect:
#     mysql -uroot
#
# To have launchd start mysql at login:
#     ln -sfv /usr/local/opt/mysql/*.plist ~/Library/LaunchAgents
# Then to load mysql now:
#     launchctl load ~/Library/LaunchAgents/homebrew.mxcl.mysql.plist
# Or, if you don't want/need launchctl, you can just run:
#     mysql.server start
# ==> /usr/local/Cellar/mysql/5.6.17_1/bin/mysql_install_db --verbose --user=quasiben --basedir=/usr/local/Cellar/mysql/5.6.17_1 --datadir=/usr/local/var/mysql --tmpdir=/tmp
# ==> Summary
# /usr/local/Cellar/mysql/5.6.17_1: 9510 files, 338M
# mysql.server restart

# unset TMPDIR
# mysql_install_db --verbose --user=`whoami` --basedir="$(brew --prefix mysql)" --datadir=/usr/local/var/mysql --tmpdir=/tmp
# sudo chown -R _mysql /usr/local/var/mysql
# sudo chmod -R o+rwx /usr/local/var/mysql
# conda install -c https://conda.binstar.org/chuongdo mysql-python

import pytest

psycopg2 = pytest.importorskip('psycopg2')
import subprocess
ps = subprocess.Popen("ps aux | grep postgres",shell=True, stdout=subprocess.PIPE)
output = ps.stdout.read()
pytestmark = pytest.mark.skipif(len(output.split('\n')) < 6, reason="No Postgres Installation")


from blaze import SQL
from blaze import CSV
from blaze.api.into import into
import sqlalchemy
import os
import csv as csv_module
from blaze import Table
from blaze import compute
import pandas as pd
import datetime as dt
import getpass

username = getpass.getuser()
url = 'mysql://{}@localhost:3306/test'.format(username)
file_name = 'test.csv'
file_name_floats = 'test_floats.csv'

def create_csv(data, file_name):
    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)


# @pytest.fixture(scope='module')
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

    sql = SQL(url,tbl, schema=csv.schema)
    engine = sql.engine
    conn = engine.raw_connection()

    cursor = conn.cursor()
    full_path = os.path.abspath(file_name)
    load = '''LOAD DATA LOCAL INFILE '{}' INTO TABLE {} FIELDS TERMINATED BY ','
        lines terminated by '\n'
        '''.format(full_path, tbl)
    cursor.execute(load)
    conn.commit()


def test_simple_into():

    tbl = 'testtable_into_2'

    csv = CSV(file_name, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]


def test_simple_float_into():

    tbl = 'testtable_into_float'

    csv = CSV(file_name_floats, columns=['a', 'b'])
    sql = SQL(url,tbl, schema= csv.schema)

    into(sql,csv, if_exists="replace")

    assert list(sql[:, 'a']) == [1.02, 102.02, 1002.02]
    assert list(sql[:, 'b']) == [2.02, 202.02, 2002.02]

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

    csv = CSV(file_name, schema='{Name: string, RegistrationDate: date, ZipCode: int64, Consts: float64}')
    import pdb
    pdb.set_trace()
    sql = SQL(url,tbl, schema=csv.schema)
    sql.extend(csv)
    # into(sql,csv, if_exists="replace")

    df = pd.read_csv(file_name, parse_dates=['RegistrationDate'])

    assert sql[0] == csv[0]

    #implement count method
    print(len(list(sql[:])))

    # assert sql[] == csv[-1]
    for col in sql.columns:
        #need to convert to python datetime
        if col == "RegistrationDate":
            py_dates = list(df['RegistrationDate'].astype(object).values)
            py_dates = [dt.date(d.year, d.month, d.day) for d in py_dates]
            assert list(sql[:,col]) == list(csv[:,col]) == py_dates
        #handle floating point precision -- perhaps it's better to call out to assert_array_almost_equal
        elif col == 'Consts':
            print([round(val, 6) for val in df[col].values])
            import pdb
            pdb.set_trace()
            # assert list(sql[:,col]) == list(csv[:,col]) == [round(val, 6) for val in df[col].values]
        else:
            assert list(sql[:,col]) == list(csv[:,col]) == list(df[col].values)
