import os
from sqlalchemy import create_engine
import sqlalchemy as sa
from dynd import nd
import unittest

from blaze.data.sql import SQL, discover
from blaze.utils import raises
from datashape import dshape
import datashape


class SingleTestClass(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:', echo=False)

    def tearDown(self):
        pass
        # How do I clean up an engine?

    def test_setup_with_uri(self):
        dd = SQL('sqlite:///:memory:',
                 'accounts',
                 schema='{name: string, amount: int}')

    def test_can_connect(self):
        with self.engine.connect() as conn:
            assert not conn.closed
        assert conn.closed

    def test_table_creation(self):
        dd = SQL(self.engine, 'testtable',
                              schema='{name: string, amount: int}',
                              primary_key='name')
        assert self.engine.has_table('testtable')


        assert dd.table.columns.get('name').primary_key
        assert not dd.table.columns.get('amount').primary_key
        assert dd.dshape == dshape('var * {name: string, amount: int}')

        assert raises(ValueError, lambda: SQL(self.engine, 'testtable2'))


    def test_extension(self):
        dd = SQL(self.engine, 'testtable2',
                               schema='{name: string, amount: int32}',
                               primary_key='name')

        data_list = [('Alice', 100), ('Bob', 50)]
        data_dict = [{'name': name, 'amount': amount} for name, amount in data_list]

        dd.extend(data_dict)

        with self.engine.connect() as conn:
            results = conn.execute('select * from testtable2')
            self.assertEquals(list(results), data_list)


        assert list(iter(dd)) == data_list or list(iter(dd)) == data_dict
        assert (dd.as_py() == tuple(map(tuple, data_list)) or
                dd.as_py() == data_dict)


    def test_chunks(self):
        schema = '{name: string, amount: int32}'
        dd = SQL(self.engine, 'testtable3',
                              schema=schema,
                              primary_key='name')

        data_list = [('Alice', 100), ('Bob', 50), ('Charlie', 200)]
        data_dict = [{'name': name, 'amount': amount} for name, amount in data_list]
        chunk = nd.array(data_list, dtype=str(dd.dshape))

        dd.extend_chunks([chunk])

        assert list(iter(dd)) == data_list or list(iter(dd)) == data_dict

        self.assertEquals(len(list(dd.chunks(blen=2))), 2)

    def test_indexing(self):
        dd = SQL(self.engine, 'testtable',
                 schema='{name: string, amount: int, id: int}',
                 primary_key='id')

        data = [('Alice', 100, 1), ('Bob', 50, 2), ('Charlie', 200, 3)]
        dd.extend(data)

        self.assertEqual(set(dd[:, ['id', 'name']]),
                        set(((1, 'Alice'), (2, 'Bob'), (3, 'Charlie'))))
        self.assertEqual(set(dd[:, 'name']), set(('Alice', 'Bob', 'Charlie')))
        assert dd[0, 'name'] in ('Alice', 'Bob', 'Charlie')
        self.assertEqual(set(dd[:, 0]), set(dd[:, 'name']))
        self.assertEqual(set(dd[:, [1, 0]]), set(dd[:, ['amount', 'name']]))
        self.assertEqual(len(list(dd[:2, 'name'])), 2)
        self.assertEqual(set(dd[:, :]), set(data))
        self.assertEqual(set(dd[:, :2]), set(dd[:, ['name', 'amount']]))
        self.assertEqual(set(dd[:]), set(dd[:, :]))
        assert dd[0] in data

    def test_inconsistent_schemas(self):
        dd = SQL('sqlite:///:memory:',
                 'badtable',
                 schema='{name: string, amount: string}')
        dd.extend([('Alice', '100'), ('Bob', '200')])

        dd2 = SQL(dd.engine,
                 'badtable',
                 schema='{name: string, amount: int}')

        assert list(dd2) == [('Alice', 100), ('Bob', 200)]


def test_discovery():
    assert discover(sa.String()) == datashape.string
    metadata = sa.MetaData()
    s = sa.Table('accounts', metadata,
                 sa.Column('name', sa.String),
                 sa.Column('amount', sa.Integer),
                 sa.Column('timestamp', sa.DateTime, primary_key=True))

    assert discover(s) == \
            dshape('var * {name: ?string, amount: ?int32, timestamp: datetime}')


def test_discover_null_columns():
    assert dshape(discover(sa.Column('name', sa.String, nullable=True))) == \
            dshape('{name: ?string}')
    assert dshape(discover(sa.Column('name', sa.String, nullable=False))) == \
            dshape('{name: string}')


def test_discovery_engine():
    dd = SQL('sqlite:///:memory:',
             'accounts',
             schema='{name: string, amount: int}')

    dshape = discover(dd.engine, 'accounts')

    assert dshape == dd.dshape


def test_extend_empty():
    dd = SQL('sqlite:///:memory:',
             'accounts',
             schema='{name: string, amount: int}')

    assert not list(dd)
    dd.extend([])
    assert not list(dd)


def test_schema_detection():
    engine = sa.create_engine('sqlite:///:memory:')
    dd = SQL(engine,
             'accounts',
             schema='{name: string, amount: int32}')

    dd.extend([['Alice', 100], ['Bob', 200]])

    dd2 = SQL(engine, 'accounts')

    assert dd.schema == dd2.schema

    if os.path.isfile('my.db'):
        os.remove('my.db')
