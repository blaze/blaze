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
    conn = pymongo.MongoClient()
    db = conn.db
    db.tmp_collection.insert(bank)
    yield db
    db.tmp_collection.drop()
    conn.close()


def test_drop(mongo):
    drop(mongo.tmp_collection)
    assert mongo.tmp_collection.count() == 0


bank_idx = [{'name': 'Alice', 'amount': 100, 'id': 1},
            {'name': 'Alice', 'amount': 200, 'id': 2},
            {'name': 'Bob', 'amount': 100, 'id': 3},
            {'name': 'Bob', 'amount': 200, 'id': 4},
            {'name': 'Bob', 'amount': 300, 'id': 5}]


@pytest.yield_fixture
def mongo_idx():
    pymongo = pytest.importorskip('pymongo')
    conn = pymongo.MongoClient()
    db = conn.db
    db.tmp_collection.insert(bank_idx)
    yield db
    db.tmp_collection.drop()
    conn.close()


class TestCreateIndex(object):
    def test_create_index(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, 'idx_id', 'id')
        assert 'idx_id' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, 'c_idx', ['id', 'amount'])
        assert 'c_idx' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index_params(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, 'c_idx',
                     [('id', ASCENDING), ('amount', DESCENDING)])
        assert 'c_idx' in mongo_idx.tmp_collection.index_information()

    def test_fails_when_using_not_list_of_tuples_or_strings(self, mongo_idx):
        with pytest.raises(AssertionError):
            create_index(mongo_idx.tmp_collection, 'asdf', [['id', DESCENDING]])

    def test_create_index_with_unique(self, mongo_idx):
        coll = mongo_idx.tmp_collection
        create_index(coll, 'c_idx', 'id', unique=True)
        assert coll.index_information()['c_idx']['unique']
