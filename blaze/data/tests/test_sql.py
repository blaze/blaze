import os
import sys
from sqlalchemy import create_engine
import sqlalchemy as sa
from dynd import nd
import unittest
import gzip

from blaze.data.sql import (discover, dshape_to_alchemy, dshape_to_table,
        create_from_datashape, into)
from blaze.sql import resource
from blaze.utils import raises, filetext, tmpfile
from datashape import dshape
import datashape
from blaze.compatibility import PY2

import pytest


class SingleTestClass(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:', echo=False)

    def tearDown(self):
        pass

    def test_can_connect(self):
        with self.engine.connect() as conn:
            assert not conn.closed
        assert conn.closed


@pytest.mark.xfail(True, reason='Can not assert type after the fact')
def test_inconsistent_schemas():
    with tmpfile('.db') as fn:
        t = resource('sqlite:///' + fn + '::badtable',
                     dshape='var * {name: string, amount: string}')
        into(t, [('Alice', '100'), ('Bob', '200')])

        t2 = resource('sqlite:///' + fn + '::badtable',
                      dshape='var * {name: string, amount: int}')

        assert into(list, t2) == [('Alice', 100), ('Bob', 200)]


def test_discovery():
    assert discover(sa.String()) == datashape.string
    metadata = sa.MetaData()
    s = sa.Table('accounts', metadata,
                 sa.Column('name', sa.String),
                 sa.Column('amount', sa.Integer),
                 sa.Column('timestamp', sa.DateTime, primary_key=True))

    assert discover(s) == \
            dshape('var * {name: ?string, amount: ?int32, timestamp: datetime}')

def test_discovery_numeric_column():
    assert discover(sa.String()) == datashape.string
    metadata = sa.MetaData()
    s = sa.Table('name', metadata,
                 sa.Column('name', sa.types.NUMERIC),)

    assert discover(s)


def test_discover_null_columns():
    assert dshape(discover(sa.Column('name', sa.String, nullable=True))) == \
            dshape('{name: ?string}')
    assert dshape(discover(sa.Column('name', sa.String, nullable=False))) == \
            dshape('{name: string}')


def single_table_engine():
    engine = sa.create_engine('sqlite:///:memory:')
    metadata = sa.MetaData(engine)
    t = sa.Table('accounts', metadata,
                 sa.Column('name', sa.String),
                 sa.Column('amount', sa.Integer))
    t.create()
    return engine, t


def test_discovery_engine():
    engine, t = single_table_engine()

    assert discover(engine, 'accounts') == discover(t)

    assert str(discover(engine)) == str(discover({'accounts': t}))


def test_extend_empty():
    engine, t = single_table_engine()

    assert not into(list, t)
    into(t, [])
    assert not into(list, t)


def test_dshape_to_alchemy():
    assert dshape_to_alchemy('string') == sa.Text
    assert isinstance(dshape_to_alchemy('string[40]'), sa.String)
    assert not isinstance(dshape_to_alchemy('string["ascii"]'), sa.Unicode)
    assert isinstance(dshape_to_alchemy('string[40, "U8"]'), sa.Unicode)
    assert dshape_to_alchemy('string[40]').length == 40

    assert dshape_to_alchemy('float32').precision == 24
    assert dshape_to_alchemy('float64').precision == 53


def test_dshape_to_table():
    t = dshape_to_table('bank', '{name: string, amount: int}')
    assert isinstance(t, sa.Table)
    assert t.name == 'bank'
    assert [c.name for c in t.c] == ['name', 'amount']


def test_create_from_datashape():
    engine = sa.create_engine('sqlite:///:memory:')
    ds = dshape('''{bank: var * {name: string, amount: int},
                    points: var * {x: int, y: int}}''')
    engine = create_from_datashape(engine, ds)

    assert discover(engine) == ds


@pytest.mark.xfail(sys.platform == 'win32' and PY2,
                   reason='Win32 py2.7 unicode/gzip/eol needs sorting out')
def test_csv_gzip_into_sql():
    from blaze.data.csv import CSV
    from blaze.data.sql import into
    engine, sql = single_table_engine()
    with filetext(b'Alice,2\nBob,4', extension='csv.gz',
                  open=gzip.open, mode='wb') as fn:
        csv = CSV(fn, schema=sql.schema)
        into(sql, csv)
        assert into(list, sql) == into(list, csv)


def test_into_table_iterator():
    engine = sa.create_engine('sqlite:///:memory:')
    metadata = sa.MetaData(engine)
    t = dshape_to_table('points', '{x: int, y: int}', metadata=metadata)
    t.create()

    data = [(1, 1), (2, 4), (3, 9)]
    into(t, data)

    assert into(list, t) == data


    t2 = dshape_to_table('points2', '{x: int, y: int}', metadata=metadata)
    t2.create()
    data2 = [{'x': 1, 'y': 1}, {'x': 2, 'y': 4}, {'x': 3, 'y': 9}]
    into(t2, data2)

    assert into(list, t2) == data


def test_extension():
    engine, t = single_table_engine()

    data_list = [('Alice', 100), ('Bob', 50)]
    data_dict = [{'name': name, 'amount': amount} for name, amount in data_list]

    into(t, data_dict)

    with engine.connect() as conn:
        results = conn.execute('select * from accounts')
        assert list(results) == data_list
