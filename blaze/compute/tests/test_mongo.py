from __future__ import absolute_import, division, print_function

import pymongo
from contextlib import contextmanager
from blaze.compute.mongo import *
from blaze.compute.core import compute
from blaze.mongo import *
from blaze.expr.table import TableSymbol
from blaze.compatibility import xfail

conn = pymongo.MongoClient()
db = conn.test_db

@contextmanager
def collection(data=[]):
    coll = db.tmp_collection
    if data:
        coll = into(coll, data)

    try:
        yield coll
    finally:
        coll.drop()


bank = [{'name': 'Alice', 'amount': 100},
        {'name': 'Alice', 'amount': 200},
        {'name': 'Bob', 'amount': 100},
        {'name': 'Bob', 'amount': 200},
        {'name': 'Bob', 'amount': 300}]

points = [{'x': 1, 'y': 10, 'z': 100},
          {'x': 2, 'y': 20, 'z': 200},
          {'x': 3, 'y': 30, 'z': 300},
          {'x': 4, 'y': 40, 'z': 400}]

t = TableSymbol('t', '{name: string, amount: int}')

p = TableSymbol('p', '{x: int, y: int, z: int}')

q = query('fake', [])

def test_tablesymbol_one():
    with collection(bank) as coll:
        assert compute_one(t, coll) == query(coll, ())

def test_tablesymbol():
    with collection(bank) as coll:
        assert compute(t, coll) == list(pluck(['name', 'amount'], bank))

def test_projection_one():
    assert compute_one(t[['name']], q).query == ({'$project': {'name': 1}},)

def test_head_one():
    assert compute_one(t.head(5), q).query == ({'$limit': 5},)

def test_head():
    with collection(bank) as coll:
        assert len(compute(t.head(2), coll)) == 2

def test_projection():
    with collection(bank) as coll:
        assert set(compute(t.name, coll)) == set(['Alice', 'Bob'])
        assert set(compute(t[['name']], coll)) == set([('Alice',), ('Bob',)])


def test_selection():
    with collection(bank) as coll:
        assert set(compute(t[t.name=='Alice'], coll)) == set([('Alice', 100),
                                                              ('Alice', 200)])
        assert set(compute(t['Alice'==t.name], coll)) == set([('Alice', 100),
                                                              ('Alice', 200)])
        assert set(compute(t[t.amount > 200], coll)) == set([('Bob', 300)])
        assert set(compute(t[t.amount >= 200], coll)) == set([('Bob', 300),
                                                              ('Bob', 200),
                                                              ('Alice', 200)])
        assert set(compute(t[t.name!='Alice'].name, coll)) == set(['Bob'])
        assert set(compute(t[(t.name=='Alice') & (t.amount > 150)], coll)) == \
                set([('Alice', 200)])
        assert set(compute(t[(t.name=='Alice') | (t.amount > 250)], coll)) == \
                set([('Alice', 200),
                     ('Alice', 100),
                     ('Bob', 300)])


@xfail()
def test_columnwise():
    with collection(points) as coll:
        assert set(compute(p.x + p.y, coll)) == set([11, 22, 33])


def test_by_one():
    assert compute_one(by(t, t.name, t.amount.sum()), q).query == \
            ({'$group': {'_id': {'name': '$name'},
                         'amount_sum': {'$sum': '$amount'}}},
             {'$project': {'amount_sum': '$amount_sum', 'name': '$_id.name'}})

def test_by():
    with collection(bank) as coll:
        assert set(compute(by(t, t.name, t.amount.sum()), coll)) == \
                set([('Alice', 300), ('Bob', 600)])
        assert set(compute(by(t, t.name, t.amount.min()), coll)) == \
                set([('Alice', 100), ('Bob', 100)])
        assert set(compute(by(t, t.name, t.amount.max()), coll)) == \
                set([('Alice', 200), ('Bob', 300)])
        assert set(compute(by(t, t.name, t.name.count()), coll)) == \
                set([('Alice', 2), ('Bob', 3)])


def test_reductions():
    with collection(bank) as coll:
        assert compute(t.amount.min(), coll) == 100
        assert compute(t.amount.max(), coll) == 300
        assert compute(t.amount.sum(), coll) == 900


def test_sort():
    with collection(bank) as coll:
        assert compute(t.amount.sort('amount'), coll) == \
                [100, 100, 200, 200, 300]
        assert compute(t.amount.sort('amount', ascending=False), coll) == \
                [300, 200, 200, 100, 100]

def test_by_multi_column():
    with collection(bank) as coll:
        assert set(compute(by(t, t[['name', 'amount']], t.count()), coll)) == \
                set([(d['name'], d['amount'], 1) for d in bank])
