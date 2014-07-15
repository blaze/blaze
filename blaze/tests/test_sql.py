
from sqlalchemy import create_engine
import sqlalchemy as sa
from blaze.sql import *
from blaze.compute.core import compute
from blaze import Table

def test_column():
    data = [(1, 2), (10, 20), (100, 200)]
    sql = SQL('sqlite:///:memory:', 'foo', schema='{x: int, y: int}')
    sql.extend(data)

    t = Table(sql)

    assert compute(t['x']) == [1, 10, 100]
    assert compute(t[['x']]) == [(1,), (10,), (100,)]

    assert compute(t.count()) == 3
