import pytest

pytest.importorskip('sqlalchemy')
import gzip
from toolz import first
import sqlalchemy
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError
from datashape import dshape
import datashape
import sys
from odo import into, drop

from blaze import create_index, resource
from blaze.sql import create_index
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

    r = list(t['x'])
    assert r == [1, 10, 100]
    assert list(t[['x']]) == [(1,), (10,), (100,)]

    assert int(t.count()) == 3


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
