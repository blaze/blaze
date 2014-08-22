import pytest
from sqlalchemy.exc import OperationalError
from blaze.sql import drop
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
