from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')
pytest.importorskip('flask.ext.cors')

from base64 import b64encode
from contextlib import contextmanager
from copy import copy

import datashape
from datashape.util.testing import assert_dshape_equal
import numpy as np
from odo import odo, convert
from datetime import datetime
from pandas import DataFrame
from pandas.util.testing import assert_frame_equal
from toolz import pipe

from blaze.dispatch import dispatch
from blaze.expr import Expr
from blaze.utils import example
from blaze import (discover, symbol, by, CSV, compute, join, into, resource,
                   Data)
from blaze.server.client import mimetype
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


class DumbResource(object):
    df = DataFrame({
        'a': np.arange(5),
        'b': np.arange(5, 10),
    })

    class NoResource(Exception):
        pass


@convert.register(DataFrame, DumbResource)
def dumb_to_df(d, return_df=None, **kwargs):
    if return_df is None:
        raise DumbResource.NoResource('return_df must be passed')
    to_return = odo(return_df, DataFrame, dshape=discover(d))
    assert_frame_equal(to_return, DumbResource.df)
    return to_return


@dispatch(Expr, DumbResource)
def compute_down(expr, d, **kwargs):
    return dumb_to_df(d, **kwargs)


@discover.register(DumbResource)
def _discover_dumb(d):
    return discover(DumbResource.df)


data = {
    'accounts': accounts,
    'cities': cities,
    'events': events,
    'db': db,
    'dumb': DumbResource(),
}


@pytest.fixture(scope='module')
def server():
    s = Server(data, all_formats)
    s.app.testing = True
    return s

@contextmanager
def temp_server(data=None):
    """For when we want to mutate the server"""
    s = Server(copy(data), formats=all_formats)
    s.app.testing = True
    yield s.app.test_client()

@pytest.yield_fixture
def test(server):
    with server.app.test_client() as c:
        yield c


@pytest.yield_fixture
def empty_server():
    s = Server(formats=all_formats)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


@pytest.yield_fixture
def iris_server():
    iris = CSV(example('iris.csv'))
    s = Server(iris, all_formats)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


def test_datasets(test):
    response = test.get('/datashape')
    assert_dshape_equal(
            datashape.dshape(response.data.decode('utf-8')),
            datashape.dshape(discover(data))
            )


@pytest.mark.parametrize('serial', all_formats)
def test_bad_responses(test, serial):
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
    expected = {
        'op': 'sum',
        'args': [
            {
                'op': 'Field',
                'args': [
                    {
                        'op': 'Symbol',
                        'args': [
                            't',
                            'var * {name: string, amount: int32}',
                            0,
                        ]
                    },
                    'amount'
                ]
            },
            [0],
            False,
        ],
    }
    assert to_tree(expr) == expected


@pytest.mark.parametrize('serial', all_formats)
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


@pytest.mark.parametrize('serial', all_formats)
def test_compute(test, serial):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    data = serial.loads(response.data)
    assert data['data'] == expected
    assert data['names'] == ['amount_sum']


@pytest.mark.parametrize('serial', all_formats)
def test_get_datetimes(test, serial):
    expr = t.events
    query = {'expr': to_tree(expr)}

    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    data = serial.loads(response.data)
    ds = datashape.dshape(data['datashape'])
    result = into(np.ndarray, data['data'], dshape=ds)
    assert into(list, result) == into(list, events)
    assert data['names'] == events.columns.tolist()


@pytest.mark.parametrize('serial', all_formats)
def dont_test_compute_with_namespace(test, serial):
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    data = serial.loads(response.data)
    assert data['data'] == expected
    assert data['names'] == ['name']


iris = CSV(example('iris.csv'))


@pytest.mark.parametrize('serial', all_formats)
def test_compute_with_variable_in_namespace(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    pl = symbol('pl', 'float32')
    expr = t[t.petal_length > pl].species
    tree = to_tree(expr, {pl: 'pl'})

    blob = serial.dumps({'expr': tree, 'namespace': {'pl': 5}})
    resp = test.post(
        '/compute',
        data=blob,
        headers=mimetype(serial)
    )

    assert 'OK' in resp.status
    data = serial.loads(resp.data)
    result = data['data']
    expected = list(compute(expr._subs({pl: 5}), {t: iris}))
    assert result == expected
    assert data['names'] == ['species']


@pytest.mark.parametrize('serial', all_formats)
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
        '/compute',
        data=blob,
        headers=mimetype(serial)
    )
    assert 'OK' in resp.status
    data = serial.loads(resp.data)
    result = DataFrame(data['data']).values
    expected = compute(expr, iris).values
    np.testing.assert_array_equal(result[:, 0], expected[:, 0])
    np.testing.assert_array_almost_equal(result[:, 1:], expected[:, 1:])
    assert data['names'] == ['species', 'max', 'sum']


@pytest.mark.parametrize('serial', all_formats)
def test_compute_column_wise(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = test.post(
        '/compute',
        data=blob,
        headers=mimetype(serial)
    )

    assert 'OK' in resp.status
    data = serial.loads(resp.data)
    result = data['data']
    expected = compute(expr, iris)
    assert list(map(tuple, result)) == into(list, expected)
    assert data['names'] == t.fields


@pytest.mark.parametrize('serial', all_formats)
def test_multi_expression_compute(test, serial):
    s = symbol('s', discover(data))

    expr = join(s.accounts, s.cities)

    resp = test.post(
        '/compute',
        data=serial.dumps(dict(expr=to_tree(expr))),
        headers=mimetype(serial)
    )

    assert 'OK' in resp.status
    respdata = serial.loads(resp.data)
    result = respdata['data']
    expected = compute(expr, {s: data})

    assert list(map(tuple, result)) == into(list, expected)
    assert respdata['names'] == expr.fields


@pytest.mark.parametrize('serial', all_formats)
def test_leaf_symbol(test, serial):
    query = {'expr': {'op': 'Field', 'args': [':leaf', 'cities']}}
    resp = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    data = serial.loads(resp.data)
    a = data['data']
    b = into(list, cities)

    assert list(map(tuple, a)) == b
    assert data['names'] == cities.columns.tolist()


@pytest.mark.parametrize('serial', all_formats)
def test_sqlalchemy_result(test, serial):
    expr = t.db.iris.head(5)
    query = {'expr': to_tree(expr)}

    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    data = serial.loads(response.data)
    result = data['data']
    assert all(isinstance(item, (tuple, list)) for item in result)
    assert data['names'] == t.db.iris.fields


def test_server_accepts_non_nonzero_ables():
    Server(DataFrame())


@pytest.mark.parametrize('serial', all_formats)
def test_server_can_compute_sqlalchemy_reductions(test, serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    respdata = serial.loads(response.data)
    result = respdata['data']
    assert result == odo(compute(expr, {t: data}), int)
    assert respdata['names'] == ['petal_length_sum']


@pytest.mark.parametrize('serial', all_formats)
def test_serialization_endpoints(test, serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post(
        '/compute',
        data=serial.dumps(query),
        headers=mimetype(serial)
    )

    assert 'OK' in response.status
    respdata = serial.loads(response.data)
    result = respdata['data']
    assert result == odo(compute(expr, {t: data}), int)
    assert respdata['names'] == ['petal_length_sum']


def test_cors_compute(test):
    res = test.options('/compute')
    assert res.status_code == 200
    assert 'HEAD' in res.headers['Allow']
    assert 'OPTIONS' in res.headers['Allow']
    assert 'POST' in res.headers['Allow']
    # we don't allow gets because we're always sending data
    assert 'GET' not in res.headers['Allow']


def test_cors_datashape(test):
    res = test.options('/datashape')
    assert res.status_code == 200
    assert 'HEAD' in res.headers['Allow']
    assert 'OPTIONS' in res.headers['Allow']
    assert 'GET' in res.headers['Allow']
    # we don't allow posts because we're just getting (meta)data.
    assert 'POST' not in res.headers['Allow']


def test_cors_add(test):
    res = test.options('/add')
    assert res.status_code == 200
    assert 'HEAD' in res.headers['Allow']
    assert 'POST' in res.headers['Allow']
    assert 'OPTIONS' in res.headers['Allow']
    # we don't allow get because we're sending data.
    assert 'GET' not in res.headers['Allow']


@pytest.fixture(scope='module')
def username():
    return 'blaze-dev'


@pytest.fixture(scope='module')
def password():
    return 'SecretPassword123'


@pytest.fixture(scope='module')
def server_with_auth(username, password):
    def auth(a):
        return a and a.username == username and a.password == password

    s = Server(data, all_formats, authorization=auth)
    s.app.testing = True
    return s


@pytest.yield_fixture
def test_with_auth(server_with_auth):
    with server_with_auth.app.test_client() as c:
        yield c


def basic_auth(username, password):
    return (
        b'Basic ' + b64encode(':'.join((username, password)).encode('utf-8'))
    )


@pytest.mark.parametrize('serial', all_formats)
def test_auth(test_with_auth, username, password, serial):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}

    r = test_with_auth.get(
        '/datashape',
        headers={'authorization': basic_auth(username, password)},
    )
    assert r.status_code == 200
    headers = mimetype(serial)
    headers['authorization'] = basic_auth(username, password)
    s = test_with_auth.post(
        '/compute',
        data=serial.dumps(query),
        headers=headers,
    )
    assert s.status_code == 200

    u = test_with_auth.get(
        '/datashape',
        headers={'authorization': basic_auth(username + 'a', password + 'a')},
    )
    assert u.status_code == 401

    headers['authorization'] = basic_auth(username + 'a', password + 'a')
    v = test_with_auth.post(
        '/compute',
        data=serial.dumps(query),
        headers=headers,
    )
    assert v.status_code == 401


@pytest.mark.parametrize('serial', all_formats)
def test_minute_query(test, serial):
    expr = t.events.when.minute
    query = {'expr': to_tree(expr)}
    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(query)
    )
    expected = {
        'data': [0, 0],
        'names': ['when_minute'],
        'datashape': '2 * int64'
    }
    assert result.status_code == 200
    assert expected == serial.loads(result.data)


@pytest.mark.parametrize('serial', all_formats)
def test_isin(test, serial):
    expr = t.events.value.isin(frozenset([1]))
    query = {'expr': to_tree(expr)}
    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(query)
    )
    expected = {
        'data': [True, False],
        'names': ['value'],
        'datashape': '2 * bool',
    }
    assert result.status_code == 200
    assert expected == serial.loads(result.data)


@pytest.mark.parametrize('serial', all_formats)
def test_add_data_to_empty_server(empty_server, serial):
    # add data
    with temp_server() as test:
        iris_path = example('iris.csv')
        blob = serial.dumps({'iris': iris_path})
        response1 = empty_server.post(
            '/add',
            headers=mimetype(serial),
            data=blob,
        )
        assert 'OK' in response1.status
        assert response1.status_code == 200

        # check for expected server datashape
        response2 = empty_server.get('/datashape')
        expected2 = str(discover({'iris': resource(iris_path)}))
        assert response2.data.decode('utf-8') == expected2

        # compute on added data
        t = Data({'iris': resource(iris_path)})
        expr = t.iris.petal_length.sum()

        response3 = empty_server.post(
            '/compute',
            data=serial.dumps({'expr': to_tree(expr)}),
            headers=mimetype(serial)
        )

        result3 = serial.loads(response3.data)['data']
        expected3 = compute(expr, {'iris': resource(iris_path)})
        assert result3 == expected3


@pytest.mark.parametrize('serial', all_formats)
def test_add_data_to_server(serial):
    with temp_server(data) as test:
        # add data
        initial_datashape = datashape.dshape(test.get('/datashape').data.decode('utf-8'))
        iris_path = example('iris.csv')
        blob = serial.dumps({'iris': iris_path})
        response1 = test.post(
            '/add',
            headers=mimetype(serial),
            data=blob,
        )
        assert 'OK' in response1.status
        assert response1.status_code == 200

        # check for expected server datashape
        new_datashape = datashape.dshape(test.get('/datashape').data.decode('utf-8'))
        data2 = data.copy()
        data2.update({'iris': resource(iris_path)})
        expected2 = datashape.dshape(discover(data2))
        from pprint import pprint as pp
        assert_dshape_equal(new_datashape, expected2)
        assert new_datashape.measure.fields != initial_datashape.measure.fields

        # compute on added data
        t = Data({'iris': resource(iris_path)})
        expr = t.iris.petal_length.sum()

        response3 = test.post(
            '/compute',
            data=serial.dumps({'expr': to_tree(expr)}),
            headers=mimetype(serial)
        )

        result3 = serial.loads(response3.data)['data']
        expected3 = compute(expr, {'iris': resource(iris_path)})
        assert result3 == expected3


@pytest.mark.parametrize('serial', all_formats)
def test_cant_add_data_to_server(iris_server, serial):
    # try adding more data to server
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': iris_path})
    response1 = iris_server.post(
        '/add',
        headers=mimetype(serial),
        data=blob,
    )
    assert response1.status_code == 422


@pytest.mark.parametrize('serial', all_formats)
def test_bad_add_payload(empty_server, serial):
    # try adding more data to server
    blob = serial.dumps('This is not a mutable mapping.')
    response1 = empty_server.post(
        '/add',
        headers=mimetype(serial),
        data=blob,
    )
    assert response1.status_code == 422


@pytest.mark.parametrize('serial', all_formats)
def test_odo_kwargs(test, serial):
    expr = t.dumb
    bad_query = {'expr': to_tree(expr)}

    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(bad_query),
    )
    assert result.status_code == 500
    assert b'return_df must be passed' in result.data

    good_query = {
        'expr': to_tree(expr),
        'odo_kwargs': {
            'return_df': odo(DumbResource.df, list),
        },
    }
    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(good_query)
    )
    assert result.status_code == 200
    data = serial.loads(result.data)
    dshape = discover(DumbResource.df)
    assert_dshape_equal(
        datashape.dshape(data['datashape']),
        dshape,
    )
    assert_frame_equal(
        odo(data['data'], DataFrame, dshape=dshape),
        DumbResource.df,
    )


@pytest.mark.parametrize('serial', all_formats)
def test_compute_kwargs(test, serial):
    expr = t.dumb.sort()
    bad_query = {'expr': to_tree(expr)}

    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(bad_query),
    )
    assert result.status_code == 500
    assert b'return_df must be passed' in result.data

    good_query = {
        'expr': to_tree(expr),
        'compute_kwargs': {
            'return_df': odo(DumbResource.df, list),
        },
    }
    result = test.post(
        '/compute',
        headers=mimetype(serial),
        data=serial.dumps(good_query)
    )
    assert result.status_code == 200
    data = serial.loads(result.data)
    dshape = discover(DumbResource.df)
    assert_dshape_equal(
        datashape.dshape(data['datashape']),
        dshape,
    )
    assert_frame_equal(
        odo(data['data'], DataFrame, dshape=dshape),
        DumbResource.df,
    )
