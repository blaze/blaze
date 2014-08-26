from __future__ import absolute_import, division, print_function

import numbers
import sys
import pytest
import numpy as np

from multipledispatch import dispatch

from blaze.compatibility import xfail
from blaze import TableSymbol, discover, dshape, compute, by, summary
from blaze import create_index, drop, RTable

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
    result = list(compute(ts.child, tb))
    assert result == list(tb.t.run(tb.conn))


@nopython3
def test_projection(ts, tb):
    result = list(compute(ts[['name', 'id']], tb))
    bank = [{'name': 'Alice', 'id': 3},
            {'name': 'Alice', 'id': 4},
            {'name': 'Bob', 'id': 7},
            {'name': 'Bob', 'id': 10},
            {'name': 'Bob', 'id': 42}]
    assert result == bank


@nopython3
def test_head_column(ts, tb):
    expr = ts.name.head(3)
    result = list(compute(expr, tb))
    assert result == [{'name': 'Alice'}, {'name': 'Alice'}, {'name': 'Bob'}]


@nopython3
def test_head(ts, tb):
    result = list(compute(ts.head(3), tb))
    assert result == [{'name': 'Alice', 'amount': 100, 'id': 3},
                      {'name': 'Alice', 'amount': 200, 'id': 4},
                      {'name': 'Bob', 'amount': 100, 'id': 7}]


@nopython3
def test_selection(ts, tb):
    q = ts[(ts.name == 'Alice') & (ts.amount < 200)]
    result = compute(q, tb)
    assert list(result) == [{'name': 'Alice', 'amount': 100, 'id': 3}]


@nopython3
def test_multiple_column_sort(ts, tb):
    expr = ts.sort(['name', 'id'], ascending=False).head(3)[['name', 'id']]
    result = list(compute(expr, tb))
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

    def test_var(self, ts, tb):
        result = compute(ts.amount.var(), tb)
        assert isinstance(result, numbers.Real)
        assert result == np.var([r['amount'] for r in bank]).item()

    @xfail(True, reason='No way to call math.sqrt on the result of var')
    def test_std(self, ts, tb):
        result = compute(ts.amount.std(), tb)
        assert isinstance(result, numbers.Real)
        assert result == np.std([r['amount'] for r in bank]).item()


@nopython3
def test_simple_by(tsc, tb):
    expr = by(tsc, tsc.name, tsc.amount.sum())
    result = compute(expr, tb)
    assert isinstance(result, dict)
    assert result == {'Alice': 300, 'Bob': 600}


@nopython3
def test_by_with_summary(tsc, tb):
    ts = tsc
    s = summary(nuniq=ts.id.nunique(), sum=ts.amount.sum(),
                mean=ts.amount.mean())
    expr = by(ts, ts.name, s)
    result = compute(expr, tb)
    assert isinstance(result, dict)
    assert result == {'Alice':
                      {'id_nuniq': 2, 'amount_sum': 300, 'amount_mean': 150},
                      'Bob':
                      {'id_nuniq': 3, 'amount_sum': 600, 'amount_mean': 200}}


@nopython3
def test_simple_summary(ts, tb):
    expr = summary(nuniq=ts.id.nunique(), sum=ts.amount.sum(),
                   mean=ts.amount.mean())
    result = compute(expr, tb)
    assert isinstance(result, dict)
    assert result == {'id_nuniq': 5, 'amount_sum': 900, 'amount_mean': 180}


@nopython3
class TestColumnWise(object):
    def test_add(self, ts, tb):
        expr = ts.id + ts.amount
        result = list(compute(expr, tb))
        assert result == [r['amount'] + r['id'] for r in bank]

    def test_nested(self, ts, tb):
        expr = 1 + ts.id + ts.amount * 2
        result = list(compute(expr, tb))
        assert result == [1 + r['id'] + r['amount'] * 2 for r in bank]

    @xfail(True, reason='ReQL does not support unary operations')
    def test_unary(self, ts, tb):
        expr = -ts.id + ts.amount
        result = list(compute(expr, tb))
        assert result == [-r['id'] + r['amount'] for r in bank]


@nopython3
def test_create_index(tb):
    create_index(tb, 'name')
    assert 'name' in tb.t.index_list().run(tb.conn)


@nopython3
@xfail(True, reason='adsf')
def test_create_index_with_name(tb):
    create_index(tb, 'name', name='awesome')


@dispatch(basestring)
def table_exists(table_name):
    import rethinkdb as rt
    return table_name in rt.table_list().run(rt.connect())


@dispatch(RTable)
def table_exists(t):
    return table_exists(t.t.args[0].data)


@pytest.fixture
def drop_tb():
    rt = pytest.importorskip('rethinkdb')
    from blaze.compute.rethink import RTable
    conn = rt.connect()
    rt.table_create('blarg').run(conn)
    t = rt.table('blarg')
    t.insert(bank).run(conn)
    return RTable(t, conn)


@nopython3
def test_drop(drop_tb):
    tb = drop_tb
    assert table_exists(tb)
    drop(tb)
    assert not table_exists(tb)
