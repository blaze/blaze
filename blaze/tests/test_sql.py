import pytest

pytest.importorskip('sqlalchemy')

from functools import partial

from toolz import first
from sqlalchemy.exc import OperationalError
from odo import into, drop

from blaze import data as bz_data
from blaze import create_index


@pytest.fixture
def sql():
    data = [(1, 2), (10, 20), (100, 200)]
    sql = bz_data(
        'sqlite:///:memory:::foo',
        dshape='var * {x: int, y: int}',
    )
    into(sql, data)
    return sql


def test_column(sql):
    t = bz_data(sql)

    r = list(t['x'])
    assert r == [1, 10, 100]
    assert list(t[['x']]) == [(1,), (10,), (100,)]

    assert int(t.count()) == 3


def test_drop(sql):
    sql = sql.data
    assert sql.exists(sql.bind)
    drop(sql)
    assert not sql.exists(sql.bind)


@pytest.mark.parametrize('cols', (
    'x', ['x'], ['y'], ['x', 'y'], ('x',), ('y',), ('x', 'y'),
))
def test_create_index(sql, cols):
    create_index(sql, cols, name='idx')
    with pytest.raises(OperationalError):
        create_index(sql, cols, name='idx')


def test_create_index_fails(sql):
    with pytest.raises(KeyError):
        create_index(sql, 'z', name='zidx')
    with pytest.raises(ValueError):
        create_index(sql, 'x')
    with pytest.raises(ValueError):
        create_index(sql, 'z')


def test_create_index_unique(sql):
    create_index(sql, 'y', name='y_idx', unique=True)
    assert len(sql.data.indexes) == 1
    idx = first(sql.data.indexes)
    assert idx.unique
    assert idx.columns.y == sql.data.c.y


def test_composite_index(sql):
    create_index(sql, ['x', 'y'], name='idx_xy')
    with pytest.raises(OperationalError):
        create_index(sql, ['x', 'y'], name='idx_xy')


def test_composite_index_fails(sql):
    with pytest.raises(KeyError):
        create_index(sql, ['z', 'bizz'], name='idx_name')


def test_composite_index_fails_with_existing_columns(sql):
    with pytest.raises(KeyError):
        create_index(sql, ['x', 'z', 'bizz'], name='idx_name')


@pytest.mark.parametrize('cols', ('x', ['x', 'y']))
def test_ignore_existing(sql, cols):
    create_call = partial(
        create_index,
        sql,
        cols,
        name='idx_name',
    )

    create_call()
    with pytest.raises(OperationalError):
        create_call(ignore_existing=False)

    # Shouldn't error
    create_call(ignore_existing=True)
