from __future__ import absolute_import, division, print_function

import pytest
import csv as csv_module
# from blaze.data import CSV, JSON
import subprocess
import tempfile
import json
import os
from blaze.utils import filetext, tmpfile, raises
from blaze.compatibility import PY3, PY2

from datashape import discover, dshape

from blaze import drop, into, create_index
from blaze.utils import assert_allclose


no_mongoimport = pytest.mark.skipif(raises(OSError,
    lambda : subprocess.Popen('mongoimport',
                              shell=os.name != 'nt',
                              stdout=subprocess.PIPE).wait()),
    reason='mongoimport cannot be found')


@pytest.yield_fixture(scope='module')
def conn():
    pymongo = pytest.importorskip('pymongo')
    try:
        c = pymongo.MongoClient()
    except pymongo.errors.ConnectionFailure:
        pytest.skip('No mongo server running')
    else:
        yield c
        c.close()


@pytest.fixture
def db(conn):
    return conn.test_db


@pytest.fixture
def tuple_data():
    return [(1, 2), (10, 20), (100, 200)]


@pytest.fixture
def openargs():
    d = {'mode': 'wb' if PY2 else 'w'}
    if PY3:
        d['newline'] = ''
    return d


@pytest.yield_fixture
def file_name_colon(tuple_data, openargs):
    with tmpfile('.csv') as filename:
        with open(filename, **openargs) as f:
            csv_module.writer(f, delimiter=':').writerows(tuple_data)
        yield filename


@pytest.yield_fixture
def file_name(tuple_data, openargs):
    with tmpfile('.csv') as filename:
        with open(filename, **openargs) as f:
            csv_module.writer(f).writerows(tuple_data)
        yield filename


@pytest.yield_fixture
def empty_collec(db):
    yield db.tmp_collection
    db.tmp_collection.drop()


@pytest.fixture
def bank():
    return [{'name': 'Alice', 'amount': 100},
            {'name': 'Alice', 'amount': 200},
            {'name': 'Bob', 'amount': 100},
            {'name': 'Bob', 'amount': 200},
            {'name': 'Bob', 'amount': 300}]


@pytest.yield_fixture
def bank_collec(db, bank):
    coll = into(db.tmp_collection, bank)
    yield coll
    coll.drop()


def test_discover(bank_collec):
    assert discover(bank_collec) == dshape('5 * {amount: int64, name: string}')


def test_into(empty_collec, bank):
    ds = dshape('var * {name: string, amount: int}')
    lhs = set(into(list, into(empty_collec, bank), dshape=ds))
    rhs = set([('Alice', 100), ('Alice', 200), ('Bob', 100), ('Bob', 200),
               ('Bob', 300)])
    assert lhs == rhs


@pytest.yield_fixture
def mongo(db, bank):
    db.tmp_collection.insert(bank)
    yield db
    db.tmp_collection.drop()


def test_drop(mongo):
    drop(mongo.tmp_collection)
    assert mongo.tmp_collection.count() == 0


@pytest.fixture
def bank_idx():
    return [{'name': 'Alice', 'amount': 100, 'id': 1},
            {'name': 'Alice', 'amount': 200, 'id': 2},
            {'name': 'Bob', 'amount': 100, 'id': 3},
            {'name': 'Bob', 'amount': 200, 'id': 4},
            {'name': 'Bob', 'amount': 300, 'id': 5}]


@pytest.yield_fixture
def mongo_idx(db, bank_idx):
    db.tmp_collection.insert(bank_idx)
    yield db
    db.tmp_collection.drop()


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
        from pymongo import ASCENDING, DESCENDING
        create_index(mongo_idx.tmp_collection,
                     [('id', ASCENDING), ('amount', DESCENDING)])
        assert 'id_1_amount_-1' in mongo_idx.tmp_collection.index_information()

    def test_fails_when_using_not_list_of_tuples_or_strings(self, mongo_idx):
        from pymongo import DESCENDING
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
        from pymongo import ASCENDING, DESCENDING
        create_index(mongo_idx.tmp_collection,
                     [('id', ASCENDING), ('amount', DESCENDING)],
                     name='c_idx')
        assert 'c_idx' in mongo_idx.tmp_collection.index_information()

    def test_fails_when_using_not_list_of_tuples_or_strings(self, mongo_idx):
        from pymongo import DESCENDING
        with pytest.raises(TypeError):
            create_index(mongo_idx.tmp_collection, [['id', DESCENDING]])

    def test_create_index_with_unique(self, mongo_idx):
        coll = mongo_idx.tmp_collection
        create_index(coll, 'id', unique=True, name='c_idx')
        assert coll.index_information()['c_idx']['unique']


@no_mongoimport
def test_csv_mongodb_load(db, file_name, empty_collec):

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

    copy_cmd = """mongoimport -d {dbname} -c {coll} --type csv --file {abspath} --fields {column_names}"""
    copy_cmd = copy_cmd.format(**copy_info)

    ps = subprocess.Popen(copy_cmd, shell=os.name != 'nt',
                          stdout=subprocess.PIPE)
    output = ps.stdout.read()
    mongo_data = list(coll.find({}, {'_0': 1, '_id': 0}))

    assert list(csv[:, '_0']) == [i['_0'] for i in mongo_data]


def test_csv_into_mongodb_colon_del(empty_collec, file_name_colon):
    csv = CSV(file_name_colon)
    coll = empty_collec
    lhs = into(list, csv)
    newcoll = into(coll, csv)
    rhs = into(list, newcoll)
    assert lhs == rhs


def test_csv_into_mongodb(empty_collec, file_name):
    csv = CSV(file_name)
    coll = empty_collec
    res = into(coll, csv)
    mongo_data = list(res.find({}, {'_0': 1, '_id': 0}))

    assert list(csv[:, '_0']) == [i['_0'] for i in mongo_data]


def test_csv_into_mongodb_columns(empty_collec, file_name):
    csv = CSV(file_name, schema='{x: int, y: int}')

    coll = empty_collec

    lhs = into(list, csv)
    assert lhs == into(list, into(coll, csv))


def test_csv_into_mongodb_complex(empty_collec):

    this_dir = os.path.dirname(__file__)
    file_name = os.path.join(this_dir, 'dummydata.csv')

    s = "{Name: string, RegistrationDate: ?datetime, ZipCode: ?int64, Consts: ?float64}"
    csv = CSV(file_name, schema=s)
    coll = empty_collec
    into(coll, csv)

    mongo_data = list(coll.find({}, {'_id': 0}))

    # This assertion doesn't work due to python floating errors
    # into(list, csv) == into(list, into(coll, csv))
    assert_allclose([list(csv[0])], [[mongo_data[0][col] for col in csv.columns]])
    assert_allclose([list(csv[9])], [[mongo_data[-1][col] for col in csv.columns]])


les_mis_data = {"nodes":[{"name":"Myriel","group":1},
                         {"name":"Napoleon","group":1},
                         {"name":"Mlle.Baptistine","group":1},
                        ],
                "links":[{"source":1,"target":0,"value":1},
                         {"source":2,"target":0,"value":8},
                         {"source":3,"target":0,"value":10},
                        ],
                }


@no_mongoimport
def test_json_into_mongodb(empty_collec):

    with filetext(json.dumps(les_mis_data)) as filename:

        dd = JSON(filename)
        coll = empty_collec
        into(coll,dd)

        mongo_data = list(coll.find())

        last = mongo_data[0]['nodes'][-1]
        first = mongo_data[0]['nodes'][0]

        first = (first['group'], first['name'])
        last = (last['group'], last['name'])

        assert dd.as_py()[1][-1] == last
        assert dd.as_py()[1][0] == first


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


@no_mongoimport
def test_jsonarray_into_mongodb(empty_collec):

    filename = tempfile.mktemp(".json")
    with open(filename, "w") as f:
        json.dump(data, f)

    dd = JSON(filename, schema="3 * { id : string, name : string, "
                                "posts : var * { content : string, title : string },"
                                " tv_show : string }")
    coll = empty_collec
    into(coll,dd, json_array=True)

    mongo_data = list(coll.find({}, {'_id': 0}))

    assert mongo_data[0] == data[0]
