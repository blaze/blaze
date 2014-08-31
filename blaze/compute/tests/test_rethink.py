from __future__ import absolute_import, division, print_function

from functools import partial
from cytoolz import compose
import numbers
import sys
import pytest
import numpy as np
import pandas as pd

from blaze.compatibility import xfail
from blaze import TableSymbol, discover, dshape, compute, by, summary, into

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


@pytest.yield_fixture(scope='module')
def create_into():
    rt = pytest.importorskip('rethinkdb')
    from blaze.compute.rethink import RTable
    conn = rt.connect()
    rt.table_create('into').run(conn)
    t = rt.table('into')
    yield RTable(t, conn)
    rt.table_drop('into').run(conn)


@pytest.yield_fixture
def tb_into(create_into):
    rt = pytest.importorskip('rethinkdb')
    from blaze.compute.rethink import RTable
    conn = rt.connect()
    t = rt.table('into')
    yield RTable(t, conn)
    t.delete().run(conn)


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

    def test_std(self, ts, tb):
        result = compute(ts.amount.std(), tb)
        assert isinstance(result, numbers.Real)
        assert result == np.std([r['amount'] for r in bank]).item()


@nopython3
def test_simple_by(tsc, tb):
    expr = by(tsc.name, tsc.amount.sum())
    result = compute(expr, tb)
    assert isinstance(result, dict)
    assert result == {'Alice': 300, 'Bob': 600}


@nopython3
def test_by_with_summary(tsc, tb):
    ts = tsc
    s = summary(nuniq=ts.id.nunique(), sum=ts.amount.sum(),
                mean=ts.amount.mean())
    expr = by(ts.name, s)
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


@xfail(raises=NotImplementedError,
       reason='ReQL does not support unary operations')
def test_unary(ts, tb):
    if sys.version_info[0] >= 3:
        pytest.xfail('RethinkDB is not compatible with Python 3')
    expr = -ts.id + ts.amount
    result = list(compute(expr, tb))
    assert result == [-r['id'] + r['amount'] for r in bank]


@nopython3
def test_map(ts, tb):
    add_one = lambda x: x + 1
    expr = ts.amount.map(add_one, schema='{amount: int64}')
    result = compute(expr, tb)
    assert result == [{'amount': add_one(r['amount'])} for r in bank]


@nopython3
def test_map_with_columns(ts, tb):
    add_one = lambda x: x + 1
    expr = ts[['amount', 'id']].map(add_one,
                                    schema='{amount: int64, id: int64}')
    result = compute(expr, tb)
    assert result == [{'amount': add_one(r['amount']),
                       'id': add_one(r['id'])} for r in bank]


@nopython3
def test_create_index(tb):
    from blaze.compute.rethink import create_index
    create_index(tb, 'name')
    assert 'name' in tb.t.index_list().run(tb.conn)


@nopython3
def test_create_index_with_name(tb):
    from blaze.compute.rethink import create_index
    import rethinkdb as rt
    with pytest.raises(rt.RqlCompileError):
        create_index(tb, 'name', name='awesome')


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
    from blaze.compute.rethink import drop
    import rethinkdb as rt
    table_name = drop_tb.t.args[0].data
    assert table_name in rt.table_list().run(rt.connect())
    drop(drop_tb)
    assert table_name not in rt.table_list().run(rt.connect())


@nopython3
class TestInto(object):
    def test_list(self, tb, tb_into):
        expected = sorted(compute(tb))
        into(tb_into, expected)
        assert sorted(compute(tb_into)) == expected

    def test_numpy(self, tb_into):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': list('def'),
                           'c': [1.0, 2.0, 3.0]})
        recs = df.to_records(index=False)
        into(tb_into, recs)
        rec_list = map(partial(compose(dict, zip), recs.dtype.fields.keys()),
                       recs)
        result = tb_into.t.with_fields('a', 'b', 'c').run(tb_into.conn)
        assert sorted(result) == sorted(rec_list)

    def test_rql(self, tb, tb_into):
        into(tb_into, tb)
        lhs = sorted(compute(tb_into))
        rhs = sorted(compute(tb))
        assert lhs == rhs
