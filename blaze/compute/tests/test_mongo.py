from __future__ import absolute_import, division, print_function

import pymongo
from contextlib import contextmanager
from blaze.compute.mongo import *
from blaze.compute.core import compute
from blaze.mongo import *
from blaze.expr.table import TableSymbol

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

t = TableSymbol('t', '{name: string, amount: int}')

q = query('fake', [])

def test_tablesymbol_one():
    with collection(bank) as coll:
        assert compute_one(t, coll) == query(coll, ())

def test_tablesymbol():
    with collection(bank) as coll:
        assert compute(t, coll) == list(pluck(['name', 'amount'], bank))

def test_projection_one():
    assert compute_one(t[['name']], q).query == ({'$project': {'name': 1}},)

def test_projection_one():
    assert compute_one(t.head(5), q).query == ({'$limit': 5},)

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
        assert set(compute(t[(t.name=='Alice') & (t.amount > 150)], coll)) == \
                set([('Alice', 200)])
        assert set(compute(t[(t.name=='Alice') | (t.amount > 250)], coll)) == \
                set([('Alice', 200),
                     ('Alice', 100),
                     ('Bob', 300)])
