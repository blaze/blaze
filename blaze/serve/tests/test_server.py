import json

from blaze.serve.server import app, datasets
from blaze.data.python import Python
from blaze.serve.index import parse_index, emit_index

accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

datasets['accounts'] = accounts
datasets['cities'] = cities

def test_basic():
    test = app.test_client()
    assert 'OK' in test.get('/').status


def test_datasets():
    test = app.test_client()
    response = test.get('/datasets.json')
    assert 'accounts' in response.data
    assert 'cities' in response.data

    assert str(accounts.dshape) in response.data
    assert str(cities.dshape) in response.data


def test_data():
    pairs = [(0, ['Alice', 100]),
             ((0, 0), 'Alice'),
             ((0, 'name'), 'Alice'),
             ((slice(0, None), 'name'), ['Alice', 'Bob'])]

    test = app.test_client()

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
    test = app.test_client()

    assert 'OK' not in test.post('/data/accounts.json',
                                 data = json.dumps(500),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/data/non-existent-table.json',
                                 data = json.dumps(0),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/data/accounts.json').status
