from __future__ import absolute_import, division, print_function

import os
import pytest

from flask import json
from datetime import datetime
from pandas import DataFrame

import blaze
from blaze.utils import example
from blaze import discover, TableSymbol, by, CSV, compute
from blaze.server.server import Server, to_tree, from_tree
from blaze.data.python import Python
from blaze.server.index import emit_index



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


def test_to_from_json():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = t.amount.sum()

    assert from_tree(to_tree(t)).isidentical(t)


def test_to_tree():
    t = TableSymbol('t', '{name: string, amount: int32}')
    expr = t.amount.sum()
    expected = {'op': 'sum',
                'args': [{'op': 'Column',
                          'args':
                            [
                              {'op': 'TableSymbol',
                               'args': [
                                    't',
                                    'var * { name : string, amount : int32 }',
                                    False
                                    ]
                               },
                              'amount'
                            ]
                        }]
                }
    assert to_tree(expr) == expected


def test_to_from_tree_namespace():
    t = TableSymbol('t', '{name: string, amount: int32}')
    expr = t.name

    tree = to_tree(expr, names={t: 't'})
    assert tree == {'op': 'Column', 'args': ['t', 'name']}

    new = from_tree(tree, namespace={'t': t})
    assert new.isidentical(expr)


def test_compute():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = t.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post('/compute/accounts_df.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected


def test_compute_with_namespace():
    t = TableSymbol('t', '{name: string, amount: int}')
    query = {'expr': {'op': 'Column',
                      'args': ['accounts_df', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post('/compute/accounts_df.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected


@pytest.fixture
def iris_server():
    iris = CSV(example('iris.csv'))
    server = Server(datasets={'iris': iris})
    return server.app.test_client()


@pytest.fixture
def iris():
    iris_path = os.path.join(os.path.dirname(blaze.__file__), os.pardir,
                             'examples', 'data', 'iris.csv')
    return CSV(iris_path)


def test_compute_by_with_summary(iris_server, iris):
    test = iris_server
    t = TableSymbol('t', iris.dshape)
    expr = by(t.species, max=t.petal_length.max(), sum=t.petal_width.sum())
    tree = to_tree(expr)
    blob = json.dumps({'expr': tree})
    resp = test.post('/compute/iris.json', data=blob,
                     content_type='application/json')
    assert 'OK' in resp.status
    result = json.loads(resp.data)['data']
    expected = compute(expr, iris)
    assert result == list(map(list, expected))


def test_compute_column_wise(iris_server, iris):
    test = iris_server
    t = TableSymbol('t', iris.dshape)
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = json.dumps({'expr': tree})
    resp = test.post('/compute/iris.json', data=blob,
                     content_type='application/json')

    assert 'OK' in resp.status
    result = json.loads(resp.data)['data']
    expected = compute(expr, iris)
    assert list(map(tuple, result)) == list(map(tuple, expected))
