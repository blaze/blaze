import pytest
from sqlalchemy.exc import OperationalError
from cytoolz import first
from blaze.sql import drop, create_index
from blaze import compute, Table, SQL


@pytest.fixture
def sql():
    data = [(1, 2), (10, 20), (100, 200)]
    sql = SQL('sqlite:///:memory:', 'foo', schema='{x: int, y: int}')
    sql.extend(data)
    return sql


def test_column(sql):
    t = Table(sql)

    r = compute(t['x'])
    assert r == [1, 10, 100]
    assert compute(t[['x']]) == [(1,), (10,), (100,)]

    assert compute(t.count()) == 3


def test_drop(sql):
    drop(sql)
    with pytest.raises(OperationalError):
        drop(sql)


class TestCreateIndex(object):

    def test_create_index(self, sql):
        create_index(sql, 'idx', 'x')
        with pytest.raises(OperationalError):
            create_index(sql, 'idx', 'x')

    def test_create_index_fails(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, 'zidx', 'z')

    def test_create_index_unique(self, sql):
        create_index(sql, 'y_idx', 'y', unique=True)
        assert len(sql.table.indexes) == 1
        idx = first(sql.table.indexes)
        assert idx.unique
        assert idx.columns.y == sql.table.c.y

    def test_composite_index(self, sql):
        create_index(sql, 'idx_xy', ['x', 'y'])
        with pytest.raises(OperationalError):
            create_index(sql, 'idx_xy', ['x', 'y'])

    def test_composite_index_fails(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, 'idx_name', ['z', 'bizz'])

    def test_composite_index_fails_with_existing_columns(self, sql):
        with pytest.raises(AttributeError):
            create_index(sql, 'idx_name', ['x', 'z', 'bizz'])

    def test_multiple_indexes(self, sql):
        create_index(sql, {'idx_x': 'x', 'idx_y': 'y'})
        with pytest.raises(OperationalError):
            create_index(sql, {'idx_x': 'x', 'idx_y': 'y'})

    def test_multiple_indexes_fails(self, sql):
        with pytest.raises(AssertionError):
            create_index(sql, {'idx_z': 'z', 'idx_h': 'h'})

    def test_multiple_indexes_fails_with_existing_columns(self, sql):
        with pytest.raises(AssertionError):
            create_index(sql, {'idx_z': 'z', 'idx_y': 'y'})
