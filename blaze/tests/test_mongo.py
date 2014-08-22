from __future__ import absolute_import, division, print_function

import pytest
pymongo = pytest.importorskip('pymongo')

try:
    pymongo.MongoClient()
except pymongo.errors.ConnectionFailure:
    pytest.importorskip('fhskjfdskfhsf')

from datashape import discover, dshape

from blaze import drop, into, create_index

conn = pymongo.MongoClient()
db = conn.test_db

from pymongo import ASCENDING, DESCENDING


@pytest.yield_fixture
def empty_collec():
    yield db.tmp_collection
    db.tmp_collection.drop()


@pytest.yield_fixture
def bank_collec():
    coll = into(db.tmp_collection, bank)
    yield coll
    coll.drop()


bank = [{'name': 'Alice', 'amount': 100},
        {'name': 'Alice', 'amount': 200},
        {'name': 'Bob', 'amount': 100},
        {'name': 'Bob', 'amount': 200},
        {'name': 'Bob', 'amount': 300}]


def test_discover(bank_collec):
    assert discover(bank_collec) == dshape('5 * {amount: int64, name: string}')


def test_into(empty_collec):
    lhs = set(into([], into(empty_collec, bank), columns=['name', 'amount']))
    rhs = set([('Alice', 100), ('Alice', 200), ('Bob', 100), ('Bob', 200),
               ('Bob', 300)])
    assert lhs == rhs


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
