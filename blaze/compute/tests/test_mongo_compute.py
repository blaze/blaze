from __future__ import absolute_import, division, print_function

import pytest
pymongo = pytest.importorskip('pymongo')

from datetime import datetime
from toolz import pluck

from blaze import into, compute, compute_up, discover, dshape

from blaze.compute.mongo import MongoQuery
from blaze.expr import TableSymbol, by
from blaze.compatibility import xfail


@pytest.fixture(scope='module')
def conn():
    try:
        return pymongo.MongoClient()
    except pymongo.errors.ConnectionFailure:
        pytest.skip('No mongo server running')


@pytest.fixture(scope='module')
def db(conn):
    return conn.test_db


@pytest.fixture
def bank_raw():
    data = [{'name': 'Alice', 'amount': 100},
            {'name': 'Alice', 'amount': 200},
            {'name': 'Bob', 'amount': 100},
            {'name': 'Bob', 'amount': 200},
            {'name': 'Bob', 'amount': 300}]
    return data


@pytest.yield_fixture
def big_bank(db):
    data = [{'name': 'Alice', 'amount': 100, 'city': 'New York City'},
            {'name': 'Alice', 'amount': 200, 'city': 'Austin'},
            {'name': 'Bob', 'amount': 100, 'city': 'New York City'},
            {'name': 'Bob', 'amount': 200, 'city': 'New York City'},
            {'name': 'Bob', 'amount': 300, 'city': 'San Francisco'}]
    coll = db.bigbank
    coll = into(coll, data)
    yield coll
    coll.drop()


@pytest.yield_fixture
def bank(db, bank_raw):
    coll = db.tmp_collection
    coll = into(coll, bank_raw)
    yield coll
    coll.drop()


@pytest.yield_fixture
def missing_vals(db):
    data = [{'x': 1, 'z': 100},
            {'x': 2, 'y': 20, 'z': 200},
            {'x': 3, 'z': 300},
            {'x': 4, 'y': 40}]
    coll = db.tmp_collection
    coll = into(coll, data)
    yield coll
    coll.drop()


@pytest.yield_fixture
def points(db):
    data = [{'x': 1, 'y': 10, 'z': 100},
            {'x': 2, 'y': 20, 'z': 200},
            {'x': 3, 'y': 30, 'z': 300},
            {'x': 4, 'y': 40, 'z': 400}]
    coll = db.tmp_collection
    coll = into(coll, data)
    yield coll
    coll.drop()


@pytest.yield_fixture
def events(db):
    data = [{'time': datetime(2012, 1, 1, 12, 00, 00), 'x': 1},
            {'time': datetime(2012, 1, 2, 12, 00, 00), 'x': 2},
            {'time': datetime(2012, 1, 3, 12, 00, 00), 'x': 3}]
    coll = db.tmp_collection
    coll = into(coll, data)
    yield coll
    coll.drop()


@pytest.fixture
def t():
    return TableSymbol('t', '{name: string, amount: int}')


@pytest.fixture
def bigt():
    return TableSymbol('bigt', '{name: string, amount: int, city: string}')


@pytest.fixture
def p():
    return TableSymbol('p', '{x: int, y: int, z: int}')


@pytest.fixture
def e():
    return TableSymbol('e', '{time: datetime, x: int}')


@pytest.fixture
def q():
    return MongoQuery('fake', [])


def test_tablesymbol_one(t, bank):
    assert compute_up(t, bank) == MongoQuery(bank, ())


def test_tablesymbol(t, bank, bank_raw):
    assert compute(t, bank) == list(pluck(['name', 'amount'], bank_raw))


def test_projection_one(t, q):
    assert compute_up(t[['name']], q).query == ({'$project': {'name': 1}},)


def test_head_one(t, q):
    assert compute_up(t.head(5), q).query == ({'$limit': 5},)


def test_head(t, bank):
    assert len(compute(t.head(2), bank)) == 2


def test_projection(t, bank):
    assert set(compute(t.name, bank)) == set(['Alice', 'Bob'])
    assert set(compute(t[['name']], bank)) == set([('Alice',), ('Bob',)])


def test_selection(t, bank):
    assert set(compute(t[t.name=='Alice'], bank)) == set([('Alice', 100),
                                                            ('Alice', 200)])
    assert set(compute(t['Alice'==t.name], bank)) == set([('Alice', 100),
                                                            ('Alice', 200)])
    assert set(compute(t[t.amount > 200], bank)) == set([('Bob', 300)])
    assert set(compute(t[t.amount >= 200], bank)) == set([('Bob', 300),
                                                          ('Bob', 200),
                                                          ('Alice', 200)])
    assert set(compute(t[t.name!='Alice'].name, bank)) == set(['Bob'])
    assert set(compute(t[(t.name=='Alice') & (t.amount > 150)], bank)) == \
            set([('Alice', 200)])
    assert set(compute(t[(t.name=='Alice') | (t.amount > 250)], bank)) == \
            set([('Alice', 200),
                    ('Alice', 100),
                    ('Bob', 300)])


@xfail(raises=NotImplementedError,
       reason='ColumnWise not implemented for MongoDB')
def test_columnwise(p, points):
    assert set(compute(p.x + p.y, points)) == set([11, 22, 33])


def test_by_one(t, q):
    assert compute_up(by(t.name, t.amount.sum()), q).query == \
            ({'$group': {'_id': {'name': '$name'},
                         'amount_sum': {'$sum': '$amount'}}},
             {'$project': {'amount_sum': '$amount_sum', 'name': '$_id.name'}})


def test_by(t, bank):
    assert set(compute(by(t.name, t.amount.sum()), bank)) == \
            set([('Alice', 300), ('Bob', 600)])
    assert set(compute(by(t.name, t.amount.min()), bank)) == \
            set([('Alice', 100), ('Bob', 100)])
    assert set(compute(by(t.name, t.amount.max()), bank)) == \
            set([('Alice', 200), ('Bob', 300)])
    assert set(compute(by(t.name, t.name.count()), bank)) == \
            set([('Alice', 2), ('Bob', 3)])


def test_reductions(t, bank):
    assert compute(t.amount.min(), bank) == 100
    assert compute(t.amount.max(), bank) == 300
    assert compute(t.amount.sum(), bank) == 900


def test_distinct(t, bank):
    assert set(compute(t.name.distinct(), bank)) == set(['Alice', 'Bob'])


def test_sort(t, bank):
    assert compute(t.amount.sort('amount'), bank) == \
            [100, 100, 200, 200, 300]
    assert compute(t.amount.sort('amount', ascending=False), bank) == \
            [300, 200, 200, 100, 100]


def test_by_multi_column(t, bank, bank_raw):
    assert set(compute(by(t[['name', 'amount']], t.count()), bank)) == \
            set([(d['name'], d['amount'], 1) for d in bank_raw])


def test_datetime_handling(e, events):
    assert set(compute(e[e.time >= datetime(2012, 1, 2, 12, 0, 0)].x,
                        events)) == set([2, 3])
    assert set(compute(e[e.time >= "2012-01-02"].x,
                        events)) == set([2, 3])


def test_summary_kwargs(t, bank):
    expr = by(t.name, total=t.amount.sum(), avg=t.amount.mean())
    result = compute(expr, bank)
    assert result == [('Bob', 200.0, 600), ('Alice', 150.0, 300)]


def test_summary_count(t, bank):
    expr = by(t.name, how_many=t.amount.count())
    result = compute(expr, bank)
    assert result == [('Bob', 3), ('Alice', 2)]


@xfail(raises=AttributeError,
       reason='ColumnWise not implemented for MongoDB')
def test_summary_arith(t, bank):
    expr = by(t.name, add_one_and_sum=(t.amount + 1).sum())
    result = compute(expr, bank)
    assert result == [('Bob', 601), ('Alice', 301)]


def test_like(t, bank):
    bank.create_index([('name', pymongo.TEXT)])
    expr = t.like(name='*Alice*')
    result = compute(expr, bank)
    assert set(result) == set((('Alice', 100), ('Alice', 200)))


def test_like_multiple(bigt, big_bank):
    expr = bigt.like(name='*Bob*', city='*York*')
    result = compute(expr, big_bank)
    assert set(result) == set((('Bob', 100, 'New York City'),
                               ('Bob', 200, 'New York City')))


def test_like_mulitple_no_match(bigt, big_bank):
    # make sure we aren't OR-ing the matches
    expr = bigt.like(name='*York*', city='*Bob*')
    result = compute(expr, big_bank)
    assert not set(result)


def test_missing_values(p, missing_vals):
    assert discover(missing_vals).subshape[0] == \
            dshape('{x: int64, y: ?int64, z: ?int64}')

    assert set(compute(p.y, missing_vals)) == set([None, 20, None, 40])
