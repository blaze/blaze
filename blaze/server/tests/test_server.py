from __future__ import absolute_import, division, print_function

import pytest

from flask import json
from datetime import datetime
from pandas import DataFrame

from blaze.utils import example
from blaze import discover, TableSymbol, by, CSV, compute
from blaze.server.server import Server, to_tree, from_tree
from blaze.server.index import emit_index


df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])

server = Server(datasets={'accounts': df})

test = server.app.test_client()


def test_datasets():
    response = test.get('/datasets.json')
    assert json.loads(response.data) == {'accounts': str(discover(df))}


def test_bad_responses():
    assert 'OK' not in test.post('/compute/accounts.json',
                                 data = json.dumps(500),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/compute/non-existent-table.json',
                                 data = json.dumps(0),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/compute/accounts.json').status


def test_to_from_json():
    t = TableSymbol('t', '{name: string, amount: int}')
    assert from_tree(to_tree(t)).isidentical(t)


def test_to_tree():
    t = TableSymbol('t', '{name: string, amount: int32}')
    expr = t.amount.sum()
    expected = {'op': 'sum',
                'args': [{'op': 'Field',
                          'args':
                            [
                              {'op': 'Symbol',
                               'args': [
                                    't',
                                    'var * { name : string, amount : int32 }',
                                    ]
                               },
                              'amount'
                            ]
                        }, [0], False]
                }
    assert to_tree(expr) == expected


def test_to_from_tree_namespace():
    t = TableSymbol('t', '{name: string, amount: int32}')
    expr = t.name

    tree = to_tree(expr, names={t: 't'})
    assert tree == {'op': 'Field', 'args': ['t', 'name']}

    new = from_tree(tree, namespace={'t': t})
    assert new.isidentical(expr)


def test_compute():
    t = TableSymbol('t', '{name: string, amount: int}')
    expr = t.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post('/compute/accounts.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data)['data'] == expected


def test_compute_with_namespace():
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post('/compute/accounts.json',
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
    return CSV(example('iris.csv'))


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
