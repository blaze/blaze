from __future__ import absolute_import, division, print_function

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

def test_csv_postgres_load():

    data = [(1, 2), (10, 20), (100, 200)]
    file_name = 'test.csv'
    tbl = 'testtable'

    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)
    engine = sqlalchemy.create_engine('postgresql://localhost/postgres')

    if engine.has_table('testtable'):
        metadata = sqlalchemy.MetaData()
        metadata.reflect(engine)
        t = metadata.tables[tbl]
        t.drop(engine)

    csv = CSV(file_name)

    sql = SQL('postgresql://localhost/postgres',tbl, schema=csv.schema)
    engine = sql.engine
    conn = engine.raw_connection()

    cursor = conn.cursor()
    full_path = os.path.abspath(file_name)
    load = '''copy {} from '{}'(FORMAT CSV, DELIMITER ',', NULL '');'''.format(tbl, full_path)
    cursor.execute(load)
    conn.commit()


def test_into():

    file_name = 'test.csv'
    tbl = 'testtable_into'

    csv = CSV(file_name)
    sql = SQL('postgresql://localhost/postgres',tbl, schema=csv.schema)

    into(sql,csv, if_exists="replace")
    t = Table(sql)

    assert compute(t['_0']) == [1L, 10L, 100L]
    assert compute(t[['_0']]) == [(1L,), (10L,), (100L,)]
    assert compute(t['_1']) == [2L, 20L, 200L]

