from __future__ import absolute_import, division, print_function

import pytest

psycopg2 = pytest.importorskip('psycopg2')
import subprocess
ps = subprocess.Popen("ps aux | grep postgres",shell=True, stdout=subprocess.PIPE)
output = ps.stdout.read()
num_processes = len(output.splitlines())
pytestmark = pytest.mark.skipif(num_processes < 6, reason="No Postgres Installation")

from blaze import SQL, into, resource
import sqlalchemy
from contextlib import contextmanager

data = [('Alice', 1), ('Bob', 2), ('Charlie', 3)]

url = 'postgresql://localhost/postgres'
engine = sqlalchemy.create_engine(url)


@contextmanager
def existing_schema(name):
    create = sqlalchemy.schema.CreateSchema(name)
    try:
        engine.execute(create)
    except:
        pass

    try:
        yield
    finally:
        metadata = sqlalchemy.MetaData()
        metadata.reflect(engine, schema=name)
        for t in metadata.tables.values():
            t.drop(bind=engine)

        drop = sqlalchemy.schema.DropSchema(name)
        engine.execute(drop)


@contextmanager
def non_existing_schema(name):
    try:
        yield
    finally:
        metadata = sqlalchemy.MetaData()
        metadata.reflect(engine, schema=name)
        for t in metadata.tables.values():
            t.drop(bind=engine)

        try:
            drop = sqlalchemy.schema.DropSchema(name)
            engine.execute(drop)
        except:
            pass


def test_sql_schema_behavior():
    with existing_schema('mydb'):
        sql = SQL(url, 'accounts', db='mydb', schema='{name: string, value: int}')
        into(sql, data)
        assert engine.has_table('accounts', schema='mydb')

        sql2 = SQL(url, 'accounts', db='mydb')
        assert list(sql2) == data

        sql3 = SQL(url, 'mydb.accounts')
        assert list(sql2) == data


def test_sql_new_schema():
    with non_existing_schema('mydb2'):
        sql = SQL(url, 'accounts', db='mydb2', schema='{name: string, value: int}')
        into(sql, data)
        assert engine.has_table('accounts', schema='mydb2')

        sql2 = SQL(url, 'accounts', db='mydb2')
        assert list(sql2) == data


def test_resource_specifying_database_name():
    with existing_schema('mydb'):
        sql = resource(url + '::mydb.accounts', schema='{name: string, value: int}')
        assert isinstance(sql, SQL)
        assert sql.table.schema == 'mydb'
