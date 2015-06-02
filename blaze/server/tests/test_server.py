from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')

import datashape
import numpy as np
from flask import json
from datetime import datetime
from pandas import DataFrame
from toolz import pipe

from odo import odo
from blaze.utils import example
from blaze import discover, symbol, by, CSV, compute, join, into, resource
from blaze.server.server import Server, to_tree, from_tree
from blaze.server.serialization import all_formats


accounts = DataFrame([['Alice', 100], ['Bob', 200]],
                     columns=['name', 'amount'])

cities = DataFrame([['Alice', 'NYC'], ['Bob', 'LA']],
                   columns=['name', 'city'])

events = DataFrame([[1, datetime(2000, 1, 1, 12, 0, 0)],
                    [2, datetime(2000, 1, 2, 12, 0, 0)]],
                   columns=['value', 'when'])

db = resource('sqlite:///' + example('iris.db'))

data = {'accounts': accounts,
          'cities': cities,
          'events': events,
              'db': db}

server = Server(data, all_formats)

test = server.app.test_client()


def test_datasets():
    response = test.get('/datashape')
    assert response.data.decode('utf-8') == str(discover(data))


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_bad_responses(serial):
    assert 'OK' not in test.post(
        '/compute/accounts.{name}'.format(name=serial.name),
        data=serial.dumps(500),
    ).status
    assert 'OK' not in test.post(
        '/compute/non-existent-table.{name}'.format(name=serial.name),
        data=serial.dumps(0),
    ).status
    assert 'OK' not in test.post(
        '/compute/accounts.{name}'.format(name=serial.name),
    ).status


def test_to_from_json():
    t = symbol('t', 'var * {name: string, amount: int}')
    assert from_tree(to_tree(t)).isidentical(t)
    assert from_tree(to_tree(t.amount + 1)).isidentical(t.amount + 1)


def test_to_tree():
    t = symbol('t', 'var * {name: string, amount: int32}')
    expr = t.amount.sum()
    expected = {'op': 'sum',
                'args': [{'op': 'Field',
                          'args':
                            [
                              {'op': 'Symbol',
                               'args': [
                                    't',
                                    'var * {name: string, amount: int32}',
                                    None
                                    ]
                               },
                              'amount'
                            ]
                        }, [0], False]
                }
    assert to_tree(expr) == expected


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_to_tree_slice(serial):
    t = symbol('t', 'var * {name: string, amount: int32}')
    expr = t[:5]
    expr2 = pipe(expr, to_tree, serial.dumps, serial.loads, from_tree)
    assert expr.isidentical(expr2)


def test_to_from_tree_namespace():
    t = symbol('t', 'var * {name: string, amount: int32}')
    expr = t.name

    tree = to_tree(expr, names={t: 't'})
    assert tree == {'op': 'Field', 'args': ['t', 'name']}

    new = from_tree(tree, namespace={'t': t})
    assert new.isidentical(expr)


def test_from_tree_is_robust_to_unnecessary_namespace():
    t = symbol('t', 'var * {name: string, amount: int32}')
    expr = t.amount + 1

    tree = to_tree(expr)  # don't use namespace

    assert from_tree(tree, {'t': t}).isidentical(expr)


t = symbol('t', discover(data))


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_compute(serial):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )

    assert 'OK' in response.status
    assert serial.loads(response.data)['data'] == expected


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_get_datetimes(serial):
    expr = t.events
    query = {'expr': to_tree(expr)}

    response = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )

    assert 'OK' in response.status
    data = serial.loads(response.data)
    ds = datashape.dshape(data['datashape'])
    result = into(np.ndarray, data['data'], dshape=ds)
    assert into(list, result) == into(list, events)


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def dont_test_compute_with_namespace(serial):
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post(
        '/compute/accounts.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )

    assert 'OK' in response.status
    assert serial.loads(response.data)['data'] == expected


@pytest.fixture
def iris_server(request):
    iris = CSV(example('iris.csv'))
    return Server(iris, all_formats).app.test_client()


iris = CSV(example('iris.csv'))


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_compute_with_variable_in_namespace(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    pl = symbol('pl', 'float32')
    expr = t[t.petal_length > pl].species
    tree = to_tree(expr, {pl: 'pl'})

    blob = serial.dumps({'expr': tree, 'namespace': {'pl': 5}})
    resp = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=blob,
    )

    assert 'OK' in resp.status
    result = serial.loads(resp.data)['data']
    expected = list(compute(expr._subs({pl: 5}), {t: iris}))
    assert result == expected


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_compute_by_with_summary(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    expr = by(
        t.species,
        max=t.petal_length.max(),
        sum=t.petal_width.sum(),
    )
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=blob,
    )
    assert 'OK' in resp.status
    result = DataFrame(serial.loads(resp.data)['data']).values
    expected = compute(expr, iris).values
    np.testing.assert_array_equal(result[:, 0], expected[:, 0])
    np.testing.assert_array_almost_equal(result[:, 1:], expected[:, 1:])


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_compute_column_wise(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=blob,
    )

    assert 'OK' in resp.status
    result = serial.loads(resp.data)['data']
    expected = compute(expr, iris)
    assert list(map(tuple, result)) == into(list, expected)


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_multi_expression_compute(serial):
    s = symbol('s', discover(data))

    expr = join(s.accounts, s.cities)

    resp = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps({'expr': to_tree(expr)}),
    )

    assert 'OK' in resp.status
    result = serial.loads(resp.data)['data']
    expected = compute(expr, {s: data})

    assert list(map(tuple, result)) == into(list, expected)


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_leaf_symbol(serial):
    query = {'expr': {'op': 'Field', 'args': [':leaf', 'cities']}}
    resp = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )

    a = serial.loads(resp.data)['data']
    b = into(list, cities)

    assert list(map(tuple, a)) == b


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_sqlalchemy_result(serial):
    expr = t.db.iris.head(5)
    query = {'expr': to_tree(expr)}

    response = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )

    assert 'OK' in response.status
    result = serial.loads(response.data)['data']
    assert all(isinstance(item, (tuple, list)) for item in result)


def test_server_accepts_non_nonzero_ables():
    Server(DataFrame())


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_server_can_compute_sqlalchemy_reductions(serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )
    assert 'OK' in response.status
    result = serial.loads(response.data)['data']
    assert result == odo(compute(expr, {t: data}), int)


@pytest.mark.parametrize(
    'serial',
    all_formats,
)
def test_serialization_endpoints(serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post(
        '/compute.{name}'.format(name=serial.name),
        data=serial.dumps(query),
    )
    assert 'OK' in response.status
    result = serial.loads(response.data)['data']
    assert result == odo(compute(expr, {t: data}), int)
