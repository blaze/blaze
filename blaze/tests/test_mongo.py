from __future__ import absolute_import, division, print_function

import pytest
import csv as csv_module
from blaze import CSV, JSON
import subprocess
import tempfile
import json
import os
pymongo = pytest.importorskip('pymongo')

try:
    pymongo.MongoClient()
except pymongo.errors.ConnectionFailure:
    pytest.importorskip('fhskjfdskfhsf')

from datashape import discover, dshape

from blaze import drop, into, create_index

conn = pymongo.MongoClient()
db = conn.test_db
file_name = 'test.csv'

from pymongo import ASCENDING, DESCENDING

def setup_function(function):
    data = [(1, 2), (10, 20), (100, 200)]

    with open(file_name, 'w') as f:
        csv_writer = csv_module.writer(f)
        for row in data:
            csv_writer.writerow(row)


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
        create_index(mongo_idx.tmp_collection, 'id')
        assert 'id_1' in mongo_idx.tmp_collection.index_information()

    def test_create_index_single_element_list(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, ['id'])
        assert 'id_1' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, ['id', 'amount'])
        assert 'id_1_amount_1' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index_params(self, mongo_idx):
        create_index(mongo_idx.tmp_collection,
                     [('id', ASCENDING), ('amount', DESCENDING)])
        assert 'id_1_amount_-1' in mongo_idx.tmp_collection.index_information()

    def test_fails_when_using_not_list_of_tuples_or_strings(self, mongo_idx):
        with pytest.raises(TypeError):
            create_index(mongo_idx.tmp_collection, [['id', DESCENDING]])

    def test_create_index_with_unique(self, mongo_idx):
        coll = mongo_idx.tmp_collection
        create_index(coll, 'id', unique=True)
        assert coll.index_information()['id_1']['unique']


class TestCreateNamedIndex(object):
    def test_create_index(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, 'id', name='idx_id')
        assert 'idx_id' in mongo_idx.tmp_collection.index_information()

    def test_create_index_single_element_list(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, ['id'], name='idx_id')
        assert 'idx_id' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index(self, mongo_idx):
        create_index(mongo_idx.tmp_collection, ['id', 'amount'], name='c_idx')
        assert 'c_idx' in mongo_idx.tmp_collection.index_information()

    def test_create_composite_index_params(self, mongo_idx):
        create_index(mongo_idx.tmp_collection,
                     [('id', ASCENDING), ('amount', DESCENDING)],
                     name='c_idx')
        assert 'c_idx' in mongo_idx.tmp_collection.index_information()

    def test_fails_when_using_not_list_of_tuples_or_strings(self, mongo_idx):
        with pytest.raises(TypeError):
            create_index(mongo_idx.tmp_collection, [['id', DESCENDING]])

    def test_create_index_with_unique(self, mongo_idx):
        coll = mongo_idx.tmp_collection
        create_index(coll, 'id', unique=True, name='c_idx')
        assert coll.index_information()['c_idx']['unique']


def test_csv_mongodb_load(empty_collec):

    csv = CSV(file_name)

    #with out header
    # mongoimport -d test_db -c testcollection --type csv --file /Users/quasiben/test.csv --fields alpha,beta
    # with collection([]) as coll:

    # --ignoreBlanks

    coll = empty_collec
    copy_info = {
        'dbname':db.name,
        'coll': coll.name,
        'abspath': csv._abspath,
        'column_names': ','.join(csv.columns)
    }

    copy_cmd = """
                mongoimport -d {dbname} -c {coll} --type csv --file {abspath} --fields {column_names}
               """
    copy_cmd = copy_cmd.format(**copy_info)

    ps = subprocess.Popen(copy_cmd,shell=True, stdout=subprocess.PIPE)
    output = ps.stdout.read()
    a = list(coll.find({},{'_0': 1, '_id': 0}))

    assert list(csv[:,'_0']) == [i['_0'] for i in a]


def test_csv_into_mongodb(empty_collec):

    csv = CSV(file_name)


    coll = empty_collec
    into(coll,csv)
    a = list(coll.find({},{'_0': 1, '_id': 0}))

    assert list(csv[:,'_0']) == [i['_0'] for i in a]

def test_csv_into_mongodb_columns(empty_collec):

    csv = CSV(file_name, schema='{x: int, y: int}')


    coll = empty_collec
    into(coll,csv)
    a = list(coll.find({},{'x': 1, '_id': 0}))

    assert list(csv[:,'x']) == [i['x'] for i in a]


def test_json_into_mongodb(empty_collec):

    this_dir = os.path.dirname(__file__)
    filename = os.path.join(this_dir, 'les_mis.json')

    dd = JSON(filename)
    coll = empty_collec
    into(coll,dd)

    a = list(coll.find())
    print(a)
    # tuplify(dd.as_py())

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


def test_jsonarray_into_mongodb(empty_collec):

    filename = tempfile.mktemp(".json")
    with open(filename, "w") as f:
        json.dump(data, f)

    dd = JSON(filename, schema = "3 * { id : string, name : string, posts : var * { content : string, title : string }, tv_show : string }")
    coll = empty_collec
    into(coll,dd, json_array=True)


    a = list(coll.find())
    print(a)
