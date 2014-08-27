from __future__ import absolute_import, division, print_function

from flask import json
from datetime import datetime
from pandas import DataFrame
import pickle

from blaze import discover, TableSymbol
from blaze.serve.server import Server
from blaze.data.python import Python
from blaze.serve.index import parse_index, emit_index
from blaze.compute.python import compute


accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

pairs = Python([(1, 2), (2, 1), (3, 4), (4, 3)],
               schema='{x: int, y: int}')

times = Python([(1, datetime(2012, 1, 1, 12, 0, 0)),
                (2, datetime(2013, 1, 1, 12, 0, 0))],
               schema='{x: int, y: datetime}')

server = Server(datasets={'accounts': accounts,
                          'accounts_df': df,
                          'cities': cities,
                          'pairs': pairs,
                          'times': times})

test = server.app.test_client()


def test_full_response():
    py_index = (slice(0, None), 'name')
    json_index = [{'start': 0, 'stop': None}, 'name']

    response = test.post('/data/accounts.json',
                         data = json.dumps({'index': emit_index(py_index)}),
                         content_type='application/json')

    print(response.data)
    assert json.loads(response.data) == \
            {'name': 'accounts',
             'datashape': "var * string",
             'index': json_index,
             'data': ['Alice', 'Bob']}


def test_datasets():
    response = test.get('/datasets.json')
    assert json.loads(response.data) == {'accounts': str(accounts.dshape),
                                         'accounts_df': str(discover(df)),
                                         'cities': str(cities.dshape),
                                         'pairs': str(pairs.dshape),
                                         'times': str(times.dshape)}


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
            print(json.loads(response.data)['data'])
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


def test_datetimes():
    query = {'index': 1}
    expected = [2, datetime(2013, 1, 1, 12, 0, 0)]

    response = test.post('/data/times.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    result = json.loads(response.data)['data']
    assert result[0] == 2
    assert '2013' in result[1]


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

def test_pickle():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = t.amount.sum()
    query = {'pickle': pickle.dumps(expr)}
    expected = 300

    response = test.post('/pickle/accounts_df.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected
