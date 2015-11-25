from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')

from base64 import b64encode
from itertools import chain

import datashape
import numpy as np
from datetime import datetime
from pandas import DataFrame
from toolz import pipe

from odo import odo
from blaze.utils import example
from blaze import discover, symbol, by, CSV, compute, join, into, resource
from blaze.server.client import mimetype, read_data
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


def create_arg(serial):
    yield serial, None
    if serial.stream_unpacker is not None:
        yield serial, 2  # chunksize=2


chunksize_params = list(chain.from_iterable(map(create_arg, all_formats)))


def post(client, url, data, headers=None, params=None):
    return client.post(
        url,
        data=data,
        headers=headers or {},
        query_string=params or {},
    )


@pytest.fixture(scope='module')
def server():
    s = Server(data, all_formats)
    s.app.testing = True
    return s


@pytest.yield_fixture
def test(server):
    with server.app.test_client() as c:
        yield c


def test_datasets(test):
    response = test.get('/datashape')
    assert response.data.decode('utf-8') == str(discover(data))


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_bad_responses(test, serial, chunksize):
    assert 'OK' not in post(
        test,
        '/compute/accounts',
        data=serial.dumps(500),
        headers=mimetype(serial)
    ).status
    assert 'OK' not in post(
        test,
        '/compute/non-existent-table',
        data=serial.dumps(0),
        headers=mimetype(serial),
    ).status
    assert 'OK' not in post(
        test,
        '/compute/accounts',
        data=None,
        headers=mimetype(serial),
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


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_compute(test, serial, chunksize):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    data = read_data(response.data, serial, chunksize)
    if chunksize:
        assert data == expected
    else:
        assert data['data'] == expected
        assert data['names'] == ['amount_sum']


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_get_datetimes(test, serial, chunksize):
    expr = t.events
    query = {'expr': to_tree(expr)}

    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    data = read_data(response.data, serial, chunksize)
    if chunksize:
        data = {'data': data, 'datashape': discover(data)}
    else:
        assert data['names'] == events.columns.tolist()
    ds = datashape.dshape(data['datashape'])
    result = into(np.ndarray, data['data'], dshape=ds)
    assert into(list, result) == into(list, events)


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def dont_test_compute_with_namespace(test, serial, chunksize):
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    data = read_data(response.data, serial, chunksize)
    if chunksize:
        assert data == expected
    else:
        assert data['data'] == expected
        assert data['names'] == ['name']


@pytest.yield_fixture
def iris_server():
    iris = CSV(example('iris.csv'))
    s = Server(iris, all_formats)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


iris = CSV(example('iris.csv'))


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_compute_with_variable_in_namespace(iris_server, serial, chunksize):
    test = iris_server
    t = symbol('t', discover(iris))
    pl = symbol('pl', 'float32')
    expr = t[t.petal_length > pl].species
    tree = to_tree(expr, {pl: 'pl'})

    blob = serial.dumps({'expr': tree, 'namespace': {'pl': 5}})
    resp = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=blob,
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in resp.status
    data = read_data(resp.data, serial, chunksize)
    if chunksize:
        result = data
    else:
        result = data['data']
        assert data['names'] == ['species']
    expected = list(compute(expr._subs({pl: 5}), {t: iris}))
    assert result == expected


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_compute_by_with_summary(iris_server, serial, chunksize):
    test = iris_server
    t = symbol('t', discover(iris))
    expr = by(
        t.species,
        max=t.petal_length.max(),
        sum=t.petal_width.sum(),
    )
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=blob,
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )
    assert 'OK' in resp.status
    data = read_data(resp.data, serial, chunksize)
    if chunksize:
        data = {'data': data}
    else:
        assert data['names'] == ['species', 'max', 'sum']
    result = DataFrame(data['data']).values
    expected = compute(expr, iris).values
    np.testing.assert_array_equal(result[:, 0], expected[:, 0])
    np.testing.assert_array_almost_equal(result[:, 1:], expected[:, 1:])


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_compute_column_wise(iris_server, serial, chunksize):
    test = iris_server
    t = symbol('t', discover(iris))
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=blob,
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in resp.status
    data = read_data(resp.data, serial, chunksize)
    if chunksize:
        result = data
    else:
        result = data['data']
        assert data['names'] == t.fields
    expected = compute(expr, iris)
    assert list(map(tuple, result)) == into(list, expected)


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_multi_expression_compute(test, serial, chunksize):
    s = symbol('s', discover(data))

    expr = join(s.accounts, s.cities)

    resp = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(dict(expr=to_tree(expr))),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in resp.status
    respdata = read_data(resp.data, serial, chunksize)
    if chunksize:
        result = respdata
    else:
        result = respdata['data']
        assert respdata['names'] == expr.fields
    expected = compute(expr, {s: data})
    assert list(map(tuple, result)) == into(list, expected)


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_leaf_symbol(test, serial, chunksize):
    query = {'expr': {'op': 'Field', 'args': [':leaf', 'cities']}}
    resp = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    data = read_data(resp.data, serial, chunksize)
    if chunksize:
        a = data
    else:
        assert data['names'] == cities.columns.tolist()
        a = data['data']
    b = into(list, cities)
    assert list(map(tuple, a)) == b


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_sqlalchemy_result(test, serial, chunksize):
    expr = t.db.iris.head(5)
    query = {'expr': to_tree(expr)}

    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    data = read_data(response.data, serial, chunksize)
    if chunksize:
        result = data
    else:
        result = data['data']
        assert data['names'] == t.db.iris.fields
    assert all(isinstance(item, (tuple, list)) for item in result)


def test_server_accepts_non_nonzero_ables():
    Server(DataFrame())


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_server_can_compute_sqlalchemy_reductions(test, serial, chunksize):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    respdata = read_data(response.data, serial, chunksize)
    if chunksize:
        result = respdata
    else:
        result = respdata['data']
        assert respdata['names'] == ['petal_length_sum']
    assert result == odo(compute(expr, {t: data}), int)


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_serialization_endpoints(test, serial, chunksize):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(query),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert 'OK' in response.status
    respdata = read_data(response.data, serial, chunksize)
    if chunksize:
        result = respdata
    else:
        result = respdata['data']
        assert respdata['names'] == ['petal_length_sum']
    assert result == odo(compute(expr, {t: data}), int)


@pytest.fixture
def has_bokeh():
    try:
        from bokeh.server.crossdomain import crossdomain
    except ImportError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize('serial,chunksize', chunksize_params)
def test_cors_compute(test, serial, chunksize, has_bokeh):
    expr = t.db.iris.petal_length.sum()
    res = post(
        test,
        '/compute' + ('/stream' if chunksize else ''),
        data=serial.dumps(dict(expr=to_tree(expr))),
        headers=mimetype(serial),
        params={'chunksize': chunksize} if chunksize else {},
    )

    assert res.status_code == 200
    assert res.headers['Access-Control-Allow-Origin'] == '*'
    assert 'HEAD' in res.headers['Access-Control-Allow-Methods']
    assert 'OPTIONS' in res.headers['Access-Control-Allow-Methods']
    assert 'POST' in res.headers['Access-Control-Allow-Methods']

    # we don't allow gets because we're always sending data
    assert 'GET' not in res.headers['Access-Control-Allow-Methods']


@pytest.mark.parametrize('method',
                         ['get',
                          pytest.mark.xfail('head', raises=AssertionError),
                          pytest.mark.xfail('options', raises=AssertionError),
                          pytest.mark.xfail('post', raises=AssertionError)])
def test_cors_datashape(test, method, has_bokeh):
    res = getattr(test, method)('/datashape')
    assert res.status_code == 200
    assert res.headers['Access-Control-Allow-Origin'] == '*'
    assert 'HEAD' not in res.headers['Access-Control-Allow-Methods']
    assert 'OPTIONS' not in res.headers['Access-Control-Allow-Methods']
    assert 'POST' not in res.headers['Access-Control-Allow-Methods']

    # we only allow GET requests
    assert 'GET' in res.headers['Access-Control-Allow-Methods']


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
