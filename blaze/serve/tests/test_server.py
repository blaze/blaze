from __future__ import absolute_import, division, print_function

from flask import json

from blaze.serve.server import Server
from blaze.data.python import Python
from blaze.serve.index import parse_index, emit_index
from blaze.compute.python import compute


accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

pairs = Python([(1, 2), (2, 1), (3, 4), (4, 3)],
               schema='{x: int, y: int}')

server = Server(datasets={'accounts': accounts,
                          'cities': cities,
                          'pairs': pairs})

test = server.app.test_client()


def test_full_response():
    py_index = (slice(0, None), 'name')
    json_index = [{'start': 0, 'stop': None}, 'name']

    response = test.post('/data/accounts.json',
                         data = json.dumps({'index': emit_index(py_index)}),
                         content_type='application/json')

    assert json.loads(response.data) == \
            {'name': 'accounts',
             'datashape': "var * string",
             'index': json_index,
             'data': ['Alice', 'Bob']}


def test_datasets():
    response = test.get('/datasets.json')
    assert json.loads(response.data) == {'accounts': str(accounts.dshape),
                                         'cities': str(cities.dshape),
                                         'pairs': str(pairs.dshape)}


def test_data():
    pairs = [(0, ['Alice', 100]),
             ((0, 0), 'Alice'),
             ((0, 'name'), 'Alice'),
             ((slice(0, None), 'name'), ['Alice', 'Bob'])]


    for ind, expected in pairs:
        index = {'index': emit_index(ind)}

        response = test.post('/data/accounts.json',
                             data = json.dumps(index),
                             content_type='application/json')
        assert 'OK' in response.status

        if not json.loads(response.data)['data'] == expected:
            print(response.data['data'])
            print(expected)
        assert json.loads(response.data)['data'] == expected


def test_bad_responses():

    assert 'OK' not in test.post('/data/accounts.json',
                                 data = json.dumps(500),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/data/non-existent-table.json',
                                 data = json.dumps(0),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/data/accounts.json').status


def test_selection():
    query = {'selection': 'x > y'}
    expected = [[2, 1], [4, 3]]

    response = test.post('/select/pairs.json',
                         data = json.dumps(query),
                         content_type='application/json')
    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected


def test_selection_on_columns():
    query = {'selection': 'city == "LA"',
             'columns': 'name'}
    expected = ['Bob']

    response = test.post('/select/cities.json',
                         data = json.dumps(query),
                         content_type='application/json')
    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected
