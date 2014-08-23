import pytest

rt = pytest.importorskip('rethinkdb')

from blaze import TableSymbol
from blaze.compute.rethink import compute_one


bank = [{'name': 'Alice', 'amount': 100, 'id': 3},
        {'name': 'Alice', 'amount': 200, 'id': 4},
        {'name': 'Bob', 'amount': 100, 'id': 7},
        {'name': 'Bob', 'amount': 200, 'id': 10},
        {'name': 'Bob', 'amount': 300, 'id': 42}]


@pytest.yield_fixture
def tb():
    conn = rt.connect()
    rt.db('test').table_create('test').run(conn)
    t = rt.table('test')
    t.insert(bank).run(conn)
    yield t
    rt.table_drop('test').run(conn)


@pytest.fixture
def conn():
    return rt.connect()


@pytest.fixture
def ts():
    return TableSymbol('t', '{name: string, amount: int64, id: int64}')


def test_table_symbol(ts, tb):
    assert compute_one(ts, tb) is tb


def test_projection(ts, tb, conn):
    result = compute_one(ts[['name', 'id']], tb, conn)
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert sorted(result) == sorted(bank)


def test_head_column(ts, tb, conn):
    result = compute_one(ts.name.head(3), tb, conn)
    assert result == [{'name': 'Alice'}, {'name': 'Alice'}, {'name': 'Bob'}]


def test_head(ts, tb, conn):
    result = compute_one(ts.head(3), tb, conn)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3},
                      {'name': 'Alice', 'amount': 200, 'id': 4},
                      {'name': 'Bob', 'amount': 100, 'id': 7}]


def test_selection(ts, tb, conn):
    q = ts[(ts.name == 'Alice') & (ts.amount < 200)]
    result = compute_one(q, tb, conn)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3}]
