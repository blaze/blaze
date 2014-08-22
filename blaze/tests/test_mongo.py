from __future__ import absolute_import, division, print_function

import pytest
pymongo = pytest.importorskip('pymongo')

try:
    pymongo.MongoClient()
except pymongo.errors.ConnectionFailure:
    pytest.importorskip('fhskjfdskfhsf')

from datashape import discover, dshape
from contextlib import contextmanager
from toolz.curried import get

from blaze import drop, into

conn = pymongo.MongoClient()
db = conn.test_db


@contextmanager
def collection(data=None):
    if data is None:
        data = []
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

def test_discover():
    with collection(bank) as coll:
        assert discover(coll) == dshape('5 * {amount: int64, name: string}')


def test_into():
    with collection([]) as coll:
        key = get(['name', 'amount'])
        assert set(into([], into(coll, bank), columns=['name', 'amount'])) ==\
                set([('Alice', 100), ('Alice', 200), ('Bob', 100),
                     ('Bob', 200), ('Bob', 300)])


@pytest.yield_fixture
def mongo():
    pymongo = pytest.importorskip('pymongo')
    conn = pymongo.MongoClient()
    db = conn.test_db
    db.tmp_collection.insert(bank)
    yield conn
    conn.close()


def test_drop(mongo):
    db = mongo.test_db
    drop(db.tmp_collection)
    assert db.tmp_collection.count() == 0
