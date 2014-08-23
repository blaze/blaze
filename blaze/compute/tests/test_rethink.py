from __future__ import absolute_import, division, print_function

import pytest

rt = pytest.importorskip('rethinkdb')

from blaze import TableSymbol, discover, dshape
from blaze.compute.rethink import compute_one


bank = [{'name': 'Alice', 'amount': 100, 'id': 3},
        {'name': 'Alice', 'amount': 200, 'id': 4},
        {'name': 'Bob', 'amount': 100, 'id': 7},
        {'name': 'Bob', 'amount': 200, 'id': 10},
        {'name': 'Bob', 'amount': 300, 'id': 42}]

conn = rt.connect()


@pytest.yield_fixture(scope='module')
def tb():
    rt.table_create('test').run(conn)
    t = rt.table('test')
    t.insert(bank).run(conn)
    yield t
    rt.table_drop('test').run(conn)


@pytest.fixture
def ts():
    dshape = '{name: string, amount: int64, id: int64}'
    return TableSymbol('t', dshape).sort('id')


@pytest.yield_fixture(scope='module')
def bsg():
    data = [{u'id': u'90742205-0032-413b-b101-ce363ba268ef',
             u'name': u'Jean-Luc Picard',
             u'posts': [{u'content': (u"There are some words I've known "
                                      "since..."),
                         u'title': u'Civil rights'}],
             u'tv_show': u'Star Trek TNG'},
            {u'id': u'7ca1d1c3-084f-490e-8b47-2b64b60ccad5',
             u'name': u'William Adama',
             u'posts': [{u'content': u'The Cylon War is long over...',
                         u'title': u'Decommissioning speech'},
                        {u'content': u'Moments ago, this ship received...',
                         u'title': u'We are at war'},
                        {u'content': u'The discoveries of the past few days...',
                         u'title': u'The new Earth'}],
             u'tv_show': u'Battlestar Galactica'},
            {u'id': u'520df804-1c91-4300-8a8d-61c2499a8b0d',
             u'name': u'Laura Roslin',
             u'posts': [{u'content': u'I, Laura Roslin, ...',
                         u'title': u'The oath of office'},
                        {u'content': u'The Cylons have the ability...',
                         u'title': u'They look like us'}],
             u'tv_show': u'Battlestar Galactica'}]
    rt.table_create('bsg').run(conn)
    rt.table('bsg').insert(data).run(conn)
    yield rt.table('bsg')
    rt.table_drop('bsg').run(conn)


def test_discover(bsg):
    result = discover(bsg)
    expected_s = ('3 * {id: string, name: string, '
                  'posts: var * {content: string, title: string},'
                  'tv_show: string}')
    expected = dshape(expected_s)
    assert result == expected


def test_table_symbol(ts, tb):
    # get the child here because we're sorting for predictable results in other
    # tests
    assert compute_one(ts.child, tb) is tb


def test_projection(ts, tb):
    result = compute_one(ts[['name', 'id']], tb, conn)
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert result == bank


def test_head_column(ts, tb):
    expr = ts.name.head(3)
    result = compute_one(expr, tb, conn)
    assert result == [{'name': 'Alice'}, {'name': 'Alice'}, {'name': 'Bob'}]


def test_head(ts, tb):
    result = compute_one(ts.head(3), tb, conn)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3},
                      {'name': 'Alice', 'amount': 200, 'id': 4},
                      {'name': 'Bob', 'amount': 100, 'id': 7}]


def test_selection(ts, tb):
    q = ts[(ts.name == 'Alice') & (ts.amount < 200)]
    result = compute_one(q, tb)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3}]


def test_multiple_column_sort(ts, tb):
    expr = ts.sort(['name', 'id'], ascending=False).head(3)[['name', 'id']]
    result = compute_one(expr, tb, conn)
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert result == bank[:-4:-1]


class TestReductions(object):
    def test_sum(self, ts, tb):
        expr = ts.amount.sum()
        result = compute_one(expr, tb, conn)
        expected = 900
        assert result == expected

    def test_min(self, ts, tb):
        result = compute_one(ts.amount.min(), tb, conn)
        assert result == 100

    def test_max(self, ts, tb):
        result = compute_one(ts.amount.max(), tb, conn)
        assert result == 300

    def test_count(self, ts, tb):
        expr = ts.head(3).id.count()
        result = compute_one(expr, tb, conn)
        assert result == 3

    def test_mean(self, ts, tb):
        expr = ts.amount.mean()
        result = compute_one(expr, tb, conn)
        expected = 900.0 / len(bank)
        assert result == expected

    def test_nunique(self, ts, tb):
        result = compute_one(ts.amount.nunique(), tb, conn)
        assert result == 3
