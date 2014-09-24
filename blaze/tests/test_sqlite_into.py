from __future__ import absolute_import, division, print_function

import pytest

from blaze import resource, drop, into
from blaze.utils import filetext


@pytest.yield_fixture
def sql(csv):
    s = resource('sqlite:///:memory:', 'test_table', schema=csv.schema)
    yield s
    drop(s)


@pytest.yield_fixture
def csv():
    with filetext('1,2\n10,20\n100,200', '.csv') as f:
        yield resource(f, schema='{a: int32, b: int32}')


def test_simple_into(sql, csv):
    into(sql, csv, if_exists="replace")
    conn = sql.engine.raw_connection()
    cursor = conn.cursor()
    with sql.engine.begin():
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' "
                       "and name='{0}';".format(sql.tablename))

    sqlite_tbl_names = cursor.fetchall()
    assert sqlite_tbl_names[0][0] == sql.tablename

    assert list(sql[:, 'a']) == [1, 10, 100]
    assert list(sql[:, 'b']) == [2, 20, 200]
