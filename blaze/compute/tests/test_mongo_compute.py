from __future__ import absolute_import, division, print_function

import pytest
import platform
pymongo = pytest.importorskip('pymongo')

from datetime import datetime
from toolz import pluck, reduceby, groupby

from datashape import Record
from blaze import into, compute, compute_up, discover, dshape, data

from blaze.compute.mongo import MongoQuery
from blaze.expr import symbol, by, floor, ceil
from blaze.compatibility import xfail

@pytest.fixture(scope='module')
def mongo_host_port():
    import os
    return (os.environ.get('MONGO_IP', 'localhost'),
            os.environ.get('MONGO_PORT', 27017))

@pytest.fixture(scope='module')
def conn(mongo_host_port):
    host, port = mongo_host_port
    try:
        return pymongo.MongoClient(host=host, port=port)
    except pymongo.errors.ConnectionFailure:
        pytest.skip('No mongo server running')


@pytest.fixture(scope='module')
def db(conn):
    return conn.test_db


bank_raw = [{'name': 'Alice', 'amount': 100},
            {'name': 'Alice', 'amount': 200},
            {'name': 'Bob', 'amount': 100},
            {'name': 'Bob', 'amount': 200},
            {'name': 'Bob', 'amount': 300}]


@pytest.yield_fixture
def big_bank(db):
    data = [{'name': 'Alice', 'amount': 100, 'city': 'New York City'},
            {'name': 'Alice', 'amount': 200, 'city': 'Austin'},
            {'name': 'Bob', 'amount': 100, 'city': 'New York City'},
            {'name': 'Bob', 'amount': 200, 'city': 'New York City'},
            {'name': 'Bob', 'amount': 300, 'city': 'San Francisco'}]
    coll = db.bigbank
    coll = into(coll, data)
    try:
        yield coll
    finally:
        coll.drop()


@pytest.yield_fixture
def date_data(db):
    n = 3
    d = {'name': ['Alice', 'Bob', 'Joe'],
         'when': [datetime(2010, 1, 1, i) for i in [1, 2, 3]],
         'amount': [100, 200, 300],
         'id': [1, 2, 3]}
    data = [dict(zip(d.keys(), [d[k][i] for k in d.keys()]))
            for i in range(n)]
    coll = into(db.date_data, data)
    try:
        yield coll
    finally:
        coll.drop()


@pytest.yield_fixture
def bank(db):
    coll = db.bank
    coll = into(coll, bank_raw)
    try:
        yield coll
    finally:
        coll.drop()


@pytest.yield_fixture
def missing_vals(db):
    data = [{'x': 1, 'z': 100},
            {'x': 2, 'y': 20, 'z': 200},
            {'x': 3, 'z': 300},
            {'x': 4, 'y': 40}]
    coll = db.missing_vals
    coll = into(coll, data)
    try:
        yield coll
    finally:
        coll.drop()


@pytest.yield_fixture
def points(db):
    data = [{'x': 1, 'y': 10, 'z': 100},
            {'x': 2, 'y': 20, 'z': 200},
            {'x': 3, 'y': 30, 'z': 300},
            {'x': 4, 'y': 40, 'z': 400}]
    coll = db.points
    coll = into(coll, data)
    try:
        yield coll
    finally:
        coll.drop()


@pytest.yield_fixture
def events(db):
    data = [{'time': datetime(2012, 1, 1, 12, 00, 00), 'x': 1},
            {'time': datetime(2012, 1, 2, 12, 00, 00), 'x': 2},
            {'time': datetime(2012, 1, 3, 12, 00, 00), 'x': 3}]
    coll = db.events
    coll = into(coll, data)
    try:
        yield coll
    finally:
        coll.drop()


t = symbol('t', 'var * {name: string, amount: int}')
bigt = symbol('bigt', 'var * {name: string, amount: int, city: string}')
p = symbol('p', 'var * {x: int, y: int, z: int}')
e = symbol('e', 'var * {time: datetime, x: int}')


q = MongoQuery('fake', [])


def test_compute_on_db(bank, points):
    assert bank.database == points.database
    db = bank.database

    d = symbol(db.name, discover(db))
    assert (compute(d.points.x.sum(), db) ==
            sum(x['x'] for x in db.points.find()))


def test_symbol(bank):
    assert compute(t, bank) == list(pluck(['name', 'amount'], bank_raw))


def test_projection_one():
    assert compute_up(t[['name']], q).query == ({'$project': {'name': 1}},)


def test_head_one():
    assert compute_up(t.head(5), q).query == ({'$limit': 5},)


def test_head(bank):
    assert len(compute(t.head(2), bank)) == 2


def test_projection(bank):
    assert set(compute(t.name, bank)) == set(['Alice', 'Bob'])
    assert set(compute(t[['name']], bank)) == set([('Alice',), ('Bob',)])


def test_selection(bank):
    assert set(compute(t[t.name == 'Alice'], bank)) == set([('Alice', 100),
                                                            ('Alice', 200)])
    assert set(compute(t['Alice' == t.name], bank)) == set([('Alice', 100),
                                                            ('Alice', 200)])
    assert set(compute(t[t.amount > 200], bank)) == set([('Bob', 300)])
    assert set(compute(t[t.amount >= 200], bank)) == set([('Bob', 300),
                                                          ('Bob', 200),
                                                          ('Alice', 200)])
    assert set(compute(t[t.name != 'Alice'].name, bank)) == set(['Bob'])
    assert set(compute(t[(t.name == 'Alice') & (t.amount > 150)], bank)) == \
        set([('Alice', 200)])
    assert set(compute(t[(t.name == 'Alice') | (t.amount > 250)], bank)) == \
        set([('Alice', 200),
             ('Alice', 100),
             ('Bob', 300)])


def test_columnwise(points):
    assert set(compute(p.x + p.y, points)) == set([11, 22, 33, 44])


def test_columnwise_multiple_operands(points):
    expected = [x['x'] + x['y'] - x['z'] * x['x'] / 2 for x in points.find()]
    assert set(compute(p.x + p.y - p.z * p.x / 2, points)) == set(expected)


def test_arithmetic(points):
    expr = p.y // p.x
    assert set(compute(expr, points)) == set(compute(expr, points.find()))


def test_columnwise_mod(points):
    expected = [x['x'] % x['y'] - x['z'] * x['x'] / 2 + 1
                for x in points.find()]
    expr = p.x % p.y - p.z * p.x / 2 + 1
    assert set(compute(expr, points)) == set(expected)


@xfail(raises=NotImplementedError,
       reason='MongoDB does not implement certain arith ops')
def test_columnwise_pow(points):
    expected = [x['x'] ** x['y'] for x in points.find()]
    assert set(compute(p.x ** p.y, points)) == set(expected)


def test_by_one():
    assert compute_up(by(t.name, total=t.amount.sum()), q).query == \
        ({'$group': {'_id': {'name': '$name'},
                     'total': {'$sum': '$amount'}}},
         {'$project': {'total': '$total', 'name': '$_id.name'}})


def test_by(bank):
    assert set(compute(by(t.name, total=t.amount.sum()), bank)) == \
        set([('Alice', 300), ('Bob', 600)])
    assert set(compute(by(t.name, min=t.amount.min()), bank)) == \
        set([('Alice', 100), ('Bob', 100)])
    assert set(compute(by(t.name, max=t.amount.max()), bank)) == \
        set([('Alice', 200), ('Bob', 300)])
    assert set(compute(by(t.name, count=t.name.count()), bank)) == \
        set([('Alice', 2), ('Bob', 3)])


def test_reductions(bank):
    assert compute(t.amount.min(), bank) == 100
    assert compute(t.amount.max(), bank) == 300
    assert compute(t.amount.sum(), bank) == 900


def test_distinct(bank):
    assert set(compute(t.name.distinct(), bank)) == set(['Alice', 'Bob'])


def test_nunique_collection(bank):
    assert compute(t.nunique(), bank) == len(bank_raw)


def test_sort(bank):
    assert compute(t.amount.sort('amount'), bank) == \
        [100, 100, 200, 200, 300]
    assert compute(t.amount.sort('amount', ascending=False), bank) == \
        [300, 200, 200, 100, 100]


def test_by_multi_column(bank):
    assert set(compute(by(t[['name', 'amount']], count=t.count()), bank)) == \
        set([(d['name'], d['amount'], 1) for d in bank_raw])


def test_datetime_handling(events):
    assert set(compute(e[e.time >= datetime(2012, 1, 2, 12, 0, 0)].x,
                       events)) == set([2, 3])
    assert set(compute(e[e.time >= "2012-01-02"].x,
                       events)) == set([2, 3])


def test_summary_kwargs(bank):
    expr = by(t.name, total=t.amount.sum(), avg=t.amount.mean())
    result = compute(expr, bank)
    assert result == [('Bob', 200.0, 600), ('Alice', 150.0, 300)]


def test_summary_count(bank):
    expr = by(t.name, how_many=t.amount.count())
    result = compute(expr, bank)
    assert result == [('Bob', 3), ('Alice', 2)]


def test_summary_arith(bank):
    expr = by(t.name, add_one_and_sum=(t.amount + 1).sum())
    result = compute(expr, bank)
    assert result == [('Bob', 603), ('Alice', 302)]


def test_summary_arith_min(bank):
    expr = by(t.name, add_one_and_sum=(t.amount + 1).min())
    result = compute(expr, bank)
    assert result == [('Bob', 101), ('Alice', 101)]


def test_summary_arith_max(bank):
    expr = by(t.name, add_one_and_sum=(t.amount + 1).max())
    result = compute(expr, bank)
    assert result == [('Bob', 301), ('Alice', 201)]


def test_summary_complex_arith(bank):
    expr = by(t.name, arith=(100 - t.amount * 2 / 30.0).sum())
    result = compute(expr, bank)
    reducer = lambda acc, x: (100 - x['amount'] * 2 / 30.0) + acc
    expected = reduceby('name', reducer, bank.find(), 0)
    assert set(result) == set(expected.items())


def test_summary_complex_arith_multiple(bank):
    expr = by(t.name, arith=(100 - t.amount * 2 / 30.0).sum(),
              other=t.amount.mean())
    result = compute(expr, bank)
    reducer = lambda acc, x: (100 - x['amount'] * 2 / 30.0) + acc
    expected = reduceby('name', reducer, bank.find(), 0)

    mu = reduceby('name', lambda acc, x: acc + x['amount'], bank.find(), 0.0)
    values = list(mu.values())
    items = expected.items()
    counts = groupby('name', bank.find())
    items = [x + (float(v) / len(counts[x[0]]),)
             for x, v in zip(items, values)]
    assert set(result) == set(items)


def test_like(bank):
    bank.create_index([('name', pymongo.TEXT)])
    expr = t[t.name.like('*Alice*')]
    result = compute(expr, bank)
    assert set(result) == set((('Alice', 100), ('Alice', 200)))


def test_like_multiple(big_bank):
    expr = bigt[bigt.name.like('*Bob*') & bigt.city.like('*York*')]
    result = compute(expr, big_bank)
    assert set(result) == set(
        (('Bob', 100, 'New York City'), ('Bob', 200, 'New York City'))
    )


def test_like_mulitple_no_match(big_bank):
    # make sure we aren't OR-ing the matches
    expr = bigt[bigt.name.like('*York*') & bigt.city.like('*Bob*')]
    assert not set(compute(expr, big_bank))


def test_missing_values(missing_vals):
    assert discover(missing_vals).subshape[0] == \
        dshape('{x: int64, y: ?int64, z: ?int64}')

    assert set(compute(p.y, missing_vals)) == set([None, 20, None, 40])


def test_datetime_access(date_data):
    t = symbol('t',
               'var * {amount: float64, id: int64, name: string, when: datetime}')

    py_data = into(list, date_data)  # a python version of the collection

    for attr in ['day', 'minute', 'second', 'year', 'month']:
        assert list(compute(getattr(t.when, attr), date_data)) == \
            list(compute(getattr(t.when, attr), py_data))


def test_datetime_access_and_arithmetic(date_data):
    t = symbol('t',
               'var * {amount: float64, id: int64, name: string, when: datetime}')

    py_data = into(list, date_data)  # a python version of the collection

    expr = t.when.day + t.id

    assert list(compute(expr, date_data)) == list(compute(expr, py_data))


def test_floor_ceil(bank):
    t = symbol('t', discover(bank))
    assert set(compute(200 * floor(t.amount / 200), bank)) == set([0, 200])
    assert set(compute(200 * ceil(t.amount / 200), bank)) == set([200, 400])


def test_Data_construct(bank, points, mongo_host_port):
    d = data('mongodb://{}:{}/test_db'.format(*mongo_host_port))
    assert 'bank' in d.fields
    assert 'points' in d.fields
    assert isinstance(d.dshape.measure, Record)


def test_Data_construct_with_table(bank, mongo_host_port):
    d = data('mongodb://{}:{}/test_db::bank'.format(*mongo_host_port))
    assert set(d.fields) == set(('name', 'amount'))
    assert int(d.count()) == 5


def test_and_same_key(bank):
    expr = t[(t.amount > 100) & (t.amount < 300)]
    result = compute(expr, bank)
    expected = [('Alice', 200), ('Bob', 200)]
    assert result == expected


def test_interactive_dshape_works(bank, mongo_host_port):
    try:
        d = data('mongodb://{}:{}/test_db::bank'.format(*mongo_host_port))
    except pymongo.errors.ConnectionFailure:
        pytest.skip('No mongo server running')
    assert dshape(d.dshape.measure) == dshape('{amount: int64, name: string}')


@pytest.mark.xfail(raises=TypeError, reason="IsIn not yet implemented")
def test_isin_fails(bank):
    expr = t[t.amount.isin([100])]
    result = compute(expr, bank)
    assert result == compute(t[t.amount == 100], bank)
