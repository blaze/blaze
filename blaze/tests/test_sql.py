import pytest
import sqlalchemy
import sqlalchemy as sa
import gzip
from cytoolz import first
from sqlalchemy.exc import OperationalError
from datashape import dshape
import datashape
import sys

from blaze.sql import (drop, create_index, resource,
        dshape_to_alchemy, dshape_to_table, create_from_datashape, into)
from blaze import compute, Data, symbol, discover
from blaze.utils import raises, filetext, tmpfile
from blaze.compatibility import PY2


@pytest.fixture
def sql():
    data = [(1, 2), (10, 20), (100, 200)]
    sql = resource('sqlite:///:memory:', 'foo', dshape='var * {x: int, y: int}')
    into(sql, data)
    return sql


def test_column(sql):
    t = Data(sql)

    r = compute(t['x'])
    assert r == [1, 10, 100]
    assert compute(t[['x']]) == [(1,), (10,), (100,)]

    assert compute(t.count()) == 3


def test_drop(sql):
    assert sql.exists(sql.bind)
    drop(sql)
    assert not sql.exists(sql.bind)


class TestCreateIndex(object):

    def test_create_index(self, sql):
        create_index(sql, 'x', name='idx')
        with pytest.raises(OperationalError):
            create_index(sql, 'x', name='idx')

    def test_create_index_fails(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, 'z', name='zidx')
        with pytest.raises(ValueError):
            create_index(sql, 'x')
        with pytest.raises(ValueError):
            create_index(sql, 'z')

    def test_create_index_unique(self, sql):
        create_index(sql, 'y', name='y_idx', unique=True)
        assert len(sql.indexes) == 1
        idx = first(sql.indexes)
        assert idx.unique
        assert idx.columns.y == sql.c.y

    def test_composite_index(self, sql):
        create_index(sql, ['x', 'y'], name='idx_xy')
        with pytest.raises(OperationalError):
            create_index(sql, ['x', 'y'], name='idx_xy')

    def test_composite_index_fails(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, ['z', 'bizz'], name='idx_name')

    def test_composite_index_fails_with_existing_columns(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, ['x', 'z', 'bizz'], name='idx_name')


def test_resource():
    with tmpfile('.db') as fn:
        uri = 'sqlite:///' + fn
        sql = resource(uri, 'foo', dshape='var * {x: int, y: int}')
        assert isinstance(sql, sa.Table)

    with tmpfile('.db') as fn:
        uri = 'sqlite:///' + fn
        sql = resource(uri + '::' + 'foo', dshape='var * {x: int, y: int}')
        assert isinstance(sql, sa.Table)


def test_resource_to_engine():
    with tmpfile('.db') as fn:
        uri = 'sqlite:///' + fn
        r = resource(uri)
        assert isinstance(r, sa.engine.Engine)
        assert r.dialect.name == 'sqlite'


def test_computation_on_engine():
    with tmpfile('.db') as fn:
        uri = 'sqlite:///' + fn
        sql = resource(uri, 'foo', dshape='var * {x: int, y: int}')
        into(sql, [(1, 2), (10, 20)])

        r = resource(uri)
        s = symbol('s', discover(r))

        assert compute(s.foo.x.max(), r) == 10



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


def test_discovery_metadata():
    engine, t = single_table_engine()
    metadata = t.metadata
    assert str(discover(metadata)) == str(discover({'accounts': t}))


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
