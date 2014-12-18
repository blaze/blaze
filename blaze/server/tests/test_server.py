from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')

import datashape
import numpy as np
from flask import json
from datetime import datetime
from pandas import DataFrame
from toolz import pipe

from blaze.utils import example
from blaze import discover, Symbol, by, CSV, compute, join, into
from blaze.server.server import Server, to_tree, from_tree
from blaze.server.index import emit_index


accounts = DataFrame([['Alice', 100], ['Bob', 200]],
                     columns=['name', 'amount'])

cities = DataFrame([['Alice', 'NYC'], ['Bob', 'LA']],
                   columns=['name', 'city'])

events = DataFrame([[1, datetime(2000, 1, 1, 12, 0, 0)],
                    [2, datetime(2000, 1, 2, 12, 0, 0)]],
                   columns=['value', 'when'])

server = Server(datasets={'accounts': accounts,
                          'cities': cities,
                          'events': events})

test = server.app.test_client()


def test_datasets():
    response = test.get('/datasets.json')
    assert json.loads(response.data.decode('utf-8')) == \
                                      {'accounts': str(discover(accounts)),
                                         'cities': str(discover(cities)),
                                         'events': str(discover(events))}


def test_bad_responses():
    assert 'OK' not in test.post('/compute/accounts.json',
                                 data = json.dumps(500),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/compute/non-existent-table.json',
                                 data = json.dumps(0),
                                 content_type='application/json').status
    assert 'OK' not in test.post('/compute/accounts.json').status


def test_to_from_json():
    t = Symbol('t', 'var * {name: string, amount: int}')
    assert from_tree(to_tree(t)).isidentical(t)
    assert from_tree(to_tree(t.amount + 1)).isidentical(t.amount + 1)


def test_to_tree():
    t = Symbol('t', 'var * {name: string, amount: int32}')
    expr = t.amount.sum()
    expected = {'op': 'sum',
                'args': [{'op': 'Field',
                          'args':
                            [
                              {'op': 'Symbol',
                               'args': [
                                    't',
                                    'var * { name : string, amount : int32 }',
                                    None
                                    ]
                               },
                              'amount'
                            ]
                        }, [0], False]
                }
    assert to_tree(expr) == expected


def test_to_tree_slice():
    t = Symbol('t', 'var * {name: string, amount: int32}')
    expr = t[:5]
    expr2 = pipe(expr, to_tree, json.dumps, json.loads, from_tree)
    assert expr.isidentical(expr2)


def test_to_from_tree_namespace():
    t = Symbol('t', 'var * {name: string, amount: int32}')
    expr = t.name

    tree = to_tree(expr, names={t: 't'})
    assert tree == {'op': 'Field', 'args': ['t', 'name']}

    new = from_tree(tree, namespace={'t': t})
    assert new.isidentical(expr)


def test_from_tree_is_robust_to_unnecessary_namespace():
    t = Symbol('t', 'var * {name: string, amount: int32}')
    expr = t.amount + 1

    tree = to_tree(expr)  # don't use namespace

    assert from_tree(tree, {'t': t}).isidentical(expr)


def test_compute():
    t = Symbol('t', 'var * {name: string, amount: int}')
    expr = t.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post('/compute/accounts.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data.decode('utf-8'))['data'] == expected


def test_get_datetimes():
    response = test.post('/compute.json',
                         data=json.dumps({'expr': 'events'}),
                         content_type='application/json')

    assert 'OK' in response.status
    data = json.loads(response.data.decode('utf-8'))
    ds = datashape.dshape(data['datashape'])
    result = into(np.ndarray, data['data'], dshape=ds)
    assert into(list, result) == into(list, events)


def test_compute_with_namespace():
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post('/compute/accounts.json',
                         data = json.dumps(query),
                         content_type='application/json')

    assert 'OK' in response.status
    assert json.loads(response.data.decode('utf-8'))['data'] == expected


@pytest.fixture
def iris_server():
    iris = CSV(example('iris.csv'))
    server = Server(datasets={'iris': iris})
    return server.app.test_client()


iris = CSV(example('iris.csv'))


def test_compute_with_variable_in_namespace(iris_server):
    test = iris_server
    t = Symbol('t', discover(iris))
    pl = Symbol('pl', 'float32')
    expr = t[t.petal_length > pl].species
    tree = to_tree(expr, {pl: 'pl'})

    blob = json.dumps({'expr': tree, 'namespace': {'pl': 5}})
    resp = test.post('/compute/iris.json', data=blob,
                     content_type='application/json')

    assert 'OK' in resp.status
    result = json.loads(resp.data.decode('utf-8'))['data']
    expected = list(compute(expr._subs({pl: 5}), {t: iris}))
    assert result == expected


def test_compute_by_with_summary(iris_server):
    test = iris_server
    t = Symbol('t', discover(iris))
    expr = by(t.species, max=t.petal_length.max(), sum=t.petal_width.sum())
    tree = to_tree(expr)
    blob = json.dumps({'expr': tree})
    resp = test.post('/compute/iris.json', data=blob,
                     content_type='application/json')
    assert 'OK' in resp.status
    result = json.loads(resp.data.decode('utf-8'))['data']
    expected = compute(expr, iris)
    assert result == list(map(list, into(list, expected)))


def test_compute_column_wise(iris_server):
    test = iris_server
    t = Symbol('t', discover(iris))
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = json.dumps({'expr': tree})
    resp = test.post('/compute/iris.json', data=blob,
                     content_type='application/json')

    assert 'OK' in resp.status
    result = json.loads(resp.data.decode('utf-8'))['data']
    expected = compute(expr, iris)
    assert list(map(tuple, result)) == into(list, expected)


def test_multi_expression_compute():
    a = Symbol('accounts', discover(accounts))
    c = Symbol('cities', discover(cities))

    expr = join(a, c)

    resp = test.post('/compute.json',
                     data=json.dumps({'expr': to_tree(expr)}),
                     content_type='application/json')

    assert 'OK' in resp.status
    result = json.loads(resp.data.decode('utf-8'))['data']
    expected = compute(expr, {a: accounts, c: cities})

    assert list(map(tuple, result))== into(list, expected)
