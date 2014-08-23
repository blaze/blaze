from __future__ import absolute_import, division, print_function

import sys
import pytest

from blaze.compatibility import xfail
from blaze import TableSymbol, discover, dshape, compute, by

nopython3 = xfail(sys.version_info[0] >= 3,
                  reason='RethinkDB is not compatible with Python 3')

bank = [{'name': 'Alice', 'amount': 100, 'id': 3},
        {'name': 'Alice', 'amount': 200, 'id': 4},
        {'name': 'Bob', 'amount': 100, 'id': 7},
        {'name': 'Bob', 'amount': 200, 'id': 10},
        {'name': 'Bob', 'amount': 300, 'id': 42}]


@pytest.yield_fixture(scope='module')
def tb():
    rt = pytest.importorskip('rethinkdb')
    from blaze.compute.rethink import RTable
    conn = rt.connect()
    rt.table_create('test').run(conn)
    t = rt.table('test')
    t.insert(bank).run(conn)
    yield RTable(t, conn)
    rt.table_drop('test').run(conn)


@pytest.fixture
def ts():
    dshape = '{name: string, amount: int64, id: int64}'
    return TableSymbol('t', dshape).sort('id')


@pytest.fixture
def tsc():
    dshape = '{name: string, amount: int64, id: int64}'
    return TableSymbol('t', dshape)


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
    rt = pytest.importorskip('rethinkdb')
    from blaze.compute.rethink import RTable
    conn = rt.connect()
    rt.table_create('bsg').run(conn)
    rt.table('bsg').insert(data).run(conn)
    yield RTable(rt.table('bsg'), conn)
    rt.table_drop('bsg').run(conn)


@nopython3
def test_discover(bsg):
    result = discover(bsg)
    expected_s = ('3 * {id: string, name: string, '
                  'posts: var * {content: string, title: string},'
                  'tv_show: string}')
    expected = dshape(expected_s)
    assert result == expected


@nopython3
def test_table_symbol(ts, tb):
    result = compute(ts.child, tb)
    assert isinstance(result, list)
    assert result == list(tb.t.run(tb.conn))


@nopython3
def test_projection(ts, tb):
    result = compute(ts[['name', 'id']], tb)
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert isinstance(result, list)
    assert result == bank


@nopython3
def test_head_column(ts, tb):
    expr = ts.name.head(3)
    result = compute(expr, tb)
    assert isinstance(result, list)
    assert result == [{'name': 'Alice'}, {'name': 'Alice'}, {'name': 'Bob'}]


@nopython3
def test_head(ts, tb):
    result = compute(ts.head(3), tb)
    assert isinstance(result, list)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3},
                      {'name': 'Alice', 'amount': 200, 'id': 4},
                      {'name': 'Bob', 'amount': 100, 'id': 7}]


@nopython3
def test_selection(ts, tb):
    q = ts[(ts.name == 'Alice') & (ts.amount < 200)]
    result = compute(q, tb)
    assert isinstance(result, list)
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3}]


@nopython3
def test_multiple_column_sort(ts, tb):
    expr = ts.sort(['name', 'id'], ascending=False).head(3)[['name', 'id']]
    result = compute(expr, tb)
    assert isinstance(result, list)
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert result == bank[:-4:-1]


@nopython3
class TestReductions(object):
    def test_sum(self, ts, tb):
        expr = ts.amount.sum()
        result = compute(expr, tb)
        assert isinstance(result, int)
        expected = 900
        assert result == expected

    def test_min(self, ts, tb):
        result = compute(ts.amount.min(), tb)
        assert isinstance(result, int)
        assert result == 100

    def test_max(self, ts, tb):
        result = compute(ts.amount.max(), tb)
        assert isinstance(result, int)
        assert result == 300

    def test_count(self, ts, tb):
        expr = ts.head(3).id.count()
        result = compute(expr, tb)
        assert isinstance(result, int)
        assert result == 3

    def test_mean(self, ts, tb):
        expr = ts.amount.mean()
        result = compute(expr, tb)
        assert isinstance(result, (int, float))
        expected = 900.0 / len(bank)
        assert result == expected

    def test_nunique(self, ts, tb):
        result = compute(ts.amount.nunique(), tb)
        assert isinstance(result, int)
        assert result == 3


@nopython3
class TestBy(object):
    def test_simple(self, tsc, tb):
        expr = by(tsc, tsc.name, tsc.amount.sum())
        result = compute(expr, tb)
        assert isinstance(result, dict)
        assert result == {'Alice': 300, 'Bob': 600}


@nopython3
class TestColumnWise(object):
    def test_add(self, ts, tb):
        expr = ts.id + ts.amount
        result = compute(expr, tb)
        assert isinstance(result, list)
        assert result == [r['amount'] + r['id'] for r in bank]
