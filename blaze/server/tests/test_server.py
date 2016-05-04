from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')
pytest.importorskip('flask.ext.cors')

from base64 import b64encode
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
from blaze import discover, symbol, by, CSV, compute, join, into, data
from blaze.server.client import mimetype
from blaze.server.server import Server, to_tree, from_tree, RC
from blaze.server.serialization import all_formats, fastmsgpack


accounts = DataFrame([['Alice', 100], ['Bob', 200]],
                     columns=['name', 'amount'])

cities = DataFrame([['Alice', 'NYC'], ['Bob', 'LA']],
                   columns=['name', 'city'])

events = DataFrame([[1, datetime(2000, 1, 1, 12, 0, 0)],
                    [2, datetime(2000, 1, 2, 12, 0, 0)]],
                   columns=['value', 'when'])

db = data('sqlite:///' + example('iris.db'))


class DumbResource(object):
    df = DataFrame({'a': np.arange(5),
                    'b': np.arange(5, 10)})

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


tdata = {'accounts': accounts,
         'cities': cities,
         'events': events,
         'db': db,
         'dumb': DumbResource()}


@pytest.fixture(scope='module')
def server():
    s = Server(tdata, all_formats)
    s.app.testing = True
    return s


@pytest.fixture(scope='module')
def add_server():
    s = Server(tdata, all_formats, allow_add=True)
    s.app.testing = True
    return s


@pytest.yield_fixture(params=[None, tdata])
def temp_server(request):
    """For when we want to mutate the server"""
    data = request.param
    s = Server(copy(data), formats=all_formats)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


@pytest.yield_fixture(params=[None, tdata])
def temp_add_server(request):
    """For when we want to mutate the server, and also add datasets to it."""
    data = request.param
    s = Server(copy(data), formats=all_formats, allow_add=True)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


@pytest.yield_fixture
def test(server):
    with server.app.test_client() as c:
        yield c


@pytest.yield_fixture
def test_add(add_server):
    with add_server.app.test_client() as c:
        yield c


@pytest.yield_fixture
def iris_server():
    iris = CSV(example('iris.csv'))
    s = Server(iris, all_formats, allow_add=True)
    s.app.testing = True
    with s.app.test_client() as c:
        yield c


def test_datasets(test):
    response = test.get('/datashape')
    assert_dshape_equal(datashape.dshape(response.data.decode('utf-8')),
                        datashape.dshape(discover(tdata)))


@pytest.mark.parametrize('serial', all_formats)
def test_bad_responses(test, serial):

    post = test.post('/compute/accounts.{name}'.format(name=serial.name),
                     data=serial.dumps(500),)
    assert 'OK' not in post.status

    post = test.post('/compute/non-existent-table.{name}'.format(name=serial.name),
                     data=serial.dumps(0))
    assert 'OK' not in post.status

    post = test.post('/compute/accounts.{name}'.format(name=serial.name))
    assert 'OK' not in post.status


def test_to_from_json():
    t = symbol('t', 'var * {name: string, amount: int}')
    assert from_tree(to_tree(t)).isidentical(t)
    assert from_tree(to_tree(t.amount + 1)).isidentical(t.amount + 1)


def test_to_tree():
    t = symbol('t', 'var * {name: string, amount: int32}')
    expr = t.amount.sum()
    dshape = datashape.dshape('var * {name: string, amount: int32}',)
    sum_args = [{'op': 'Field',
                 'args': [{'op': 'Symbol',
                           'args': ['t', dshape, 0]},
                          'amount']},
                [0],
                False]
    expected = {'op': 'sum', 'args': sum_args}
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


t = symbol('t', discover(tdata))


@pytest.mark.parametrize('serial', all_formats)
def test_compute(test, serial):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}
    expected = 300

    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    tdata = serial.loads(response.data)
    assert serial.data_loads(tdata['data']) == expected
    assert list(tdata['names']) == ['amount_sum']


@pytest.mark.parametrize('serial', all_formats)
def test_get_datetimes(test, serial):
    expr = t.events
    query = {'expr': to_tree(expr)}

    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    tdata = serial.loads(response.data)
    ds = datashape.dshape(tdata['datashape'])
    result = into(np.ndarray,
                  serial.data_loads(tdata['data']),
                  dshape=ds)
    assert into(list, result) == into(list, events)
    assert list(tdata['names']) == events.columns.tolist()


@pytest.mark.parametrize('serial', all_formats)
def dont_test_compute_with_namespace(test, serial):
    query = {'expr': {'op': 'Field',
                      'args': ['accounts', 'name']}}
    expected = ['Alice', 'Bob']

    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    tdata = serial.loads(response.data)
    assert serial.data_loads(tdata['data']) == expected
    assert tdata['names'] == ['name']


iris = CSV(example('iris.csv'))


@pytest.mark.parametrize('serial', all_formats)
def test_compute_with_variable_in_namespace(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    pl = symbol('pl', 'float32')
    expr = t[t.petal_length > pl].species
    tree = to_tree(expr, {pl: 'pl'})

    blob = serial.dumps({'expr': tree, 'namespace': {'pl': 5}})
    resp = test.post('/compute',
                     data=blob,
                     headers=mimetype(serial))

    assert 'OK' in resp.status
    tdata = serial.loads(resp.data)
    result = serial.data_loads(tdata['data'])
    expected = list(compute(expr._subs({pl: 5}), {t: iris}))
    assert odo(result, list) == expected
    assert list(tdata['names']) == ['species']


@pytest.mark.parametrize('serial', all_formats)
def test_compute_by_with_summary(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    expr = by(t.species,
              max=t.petal_length.max(),
              sum=t.petal_width.sum())
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = test.post('/compute',
                     data=blob,
                     headers=mimetype(serial))
    assert 'OK' in resp.status
    tdata = serial.loads(resp.data)
    result = DataFrame(serial.data_loads(tdata['data'])).values
    expected = compute(expr, iris).values
    np.testing.assert_array_equal(result[:, 0],
                                  expected[:, 0])
    np.testing.assert_array_almost_equal(result[:, 1:],
                                         expected[:, 1:])
    assert list(tdata['names']) == ['species', 'max', 'sum']


@pytest.mark.parametrize('serial', all_formats)
def test_compute_column_wise(iris_server, serial):
    test = iris_server
    t = symbol('t', discover(iris))
    subexpr = ((t.petal_width / 2 > 0.5) &
               (t.petal_length / 2 > 0.5))
    expr = t[subexpr]
    tree = to_tree(expr)
    blob = serial.dumps({'expr': tree})
    resp = test.post('/compute',
                     data=blob,
                     headers=mimetype(serial))

    assert 'OK' in resp.status
    tdata = serial.loads(resp.data)
    result = serial.data_loads(tdata['data'])
    expected = compute(expr, iris)
    assert list(map(tuple, into(list, result))) == into(list, expected)
    assert list(tdata['names']) == t.fields


@pytest.mark.parametrize('serial', all_formats)
def test_multi_expression_compute(test, serial):
    s = symbol('s', discover(tdata))

    expr = join(s.accounts, s.cities)

    resp = test.post('/compute',
                     data=serial.dumps({'expr': to_tree(expr)}),
                     headers=mimetype(serial))

    assert 'OK' in resp.status
    respdata = serial.loads(resp.data)
    result = serial.data_loads(respdata['data'])
    expected = compute(expr, {s: tdata})

    assert list(map(tuple, odo(result, list))) == into(list, expected)
    assert list(respdata['names']) == expr.fields


@pytest.mark.parametrize('serial', all_formats)
def test_leaf_symbol(test, serial):
    query = {'expr': {'op': 'Field', 'args': [':leaf', 'cities']}}
    resp = test.post('/compute',
                     data=serial.dumps(query),
                     headers=mimetype(serial))

    tdata = serial.loads(resp.data)
    a = serial.data_loads(tdata['data'])
    b = into(list, cities)

    assert list(map(tuple, into(list, a))) == b
    assert list(tdata['names']) == cities.columns.tolist()


@pytest.mark.parametrize('serial', all_formats)
def test_sqlalchemy_result(test, serial):
    expr = t.db.iris.head(5)
    query = {'expr': to_tree(expr)}

    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    tdata = serial.loads(response.data)
    result = serial.data_loads(tdata['data'])
    if isinstance(result, list):
        assert all(isinstance(item, (tuple, list)) for item in result)
    elif isinstance(result, DataFrame):
        expected = DataFrame([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa'],
                              [4.9, 3.0, 1.4, 0.2, 'Iris-setosa'],
                              [4.7, 3.2, 1.3, 0.2, 'Iris-setosa'],
                              [4.6, 3.1, 1.5, 0.2, 'Iris-setosa'],
                              [5.0, 3.6, 1.4, 0.2, 'Iris-setosa']],
                             columns=['sepal_length',
                                      'sepal_width',
                                      'petal_length',
                                      'petal_width',
                                      'species'])
        assert_frame_equal(expected, result)
    assert list(tdata['names']) == t.db.iris.fields


def test_server_accepts_non_nonzero_ables():
    Server(DataFrame())


@pytest.mark.parametrize('serial', all_formats)
def test_server_can_compute_sqlalchemy_reductions(test, serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    respdata = serial.loads(response.data)
    result = serial.data_loads(respdata['data'])
    assert result == into(int, compute(expr, {t: tdata}))
    assert list(respdata['names']) == ['petal_length_sum']


@pytest.mark.parametrize('serial', all_formats)
def test_serialization_endpoints(test, serial):
    expr = t.db.iris.petal_length.sum()
    query = {'expr': to_tree(expr)}
    response = test.post('/compute',
                         data=serial.dumps(query),
                         headers=mimetype(serial))

    assert 'OK' in response.status
    respdata = serial.loads(response.data)
    result = serial.data_loads(respdata['data'])
    assert result == into(int, compute(expr, {t: tdata}))
    assert list(respdata['names']) == ['petal_length_sum']


def test_cors_compute(test):
    res = test.options('/compute')
    assert res.status_code == RC.OK
    assert 'HEAD' in res.headers['Allow']
    assert 'OPTIONS' in res.headers['Allow']
    assert 'POST' in res.headers['Allow']
    # we don't allow gets because we're always sending data
    assert 'GET' not in res.headers['Allow']


def test_cors_datashape(test):
    res = test.options('/datashape')
    assert res.status_code == RC.OK
    assert 'HEAD' in res.headers['Allow']
    assert 'OPTIONS' in res.headers['Allow']
    assert 'GET' in res.headers['Allow']
    # we don't allow posts because we're just getting (meta)data.
    assert 'POST' not in res.headers['Allow']


def test_cors_add(test_add):
    res = test_add.options('/add')
    assert res.status_code == RC.OK
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

    s = Server(tdata, all_formats, authorization=auth)
    s.app.testing = True
    return s


@pytest.yield_fixture
def test_with_auth(server_with_auth):
    with server_with_auth.app.test_client() as c:
        yield c


def basic_auth(username, password):
    return b'Basic ' + b64encode(':'.join((username, password)).encode('utf-8'))


@pytest.mark.parametrize('serial', all_formats)
def test_auth(test_with_auth, username, password, serial):
    expr = t.accounts.amount.sum()
    query = {'expr': to_tree(expr)}

    r = test_with_auth.get('/datashape',
                           headers={'authorization': basic_auth(username, password)})
    assert r.status_code == RC.OK
    headers = mimetype(serial)
    headers['authorization'] = basic_auth(username, password)
    s = test_with_auth.post('/compute',
                            data=serial.dumps(query),
                            headers=headers)
    assert s.status_code == RC.OK

    u = test_with_auth.get('/datashape',
                           headers={'authorization': basic_auth(username + 'a', password + 'a')})
    assert u.status_code == RC.UNAUTHORIZED

    headers['authorization'] = basic_auth(username + 'a', password + 'a')
    v = test_with_auth.post('/compute',
                            data=serial.dumps(query),
                            headers=headers)
    assert v.status_code == RC.UNAUTHORIZED


@pytest.mark.parametrize('serial', all_formats)
def test_minute_query(test, serial):
    expr = t.events.when.minute
    query = {'expr': to_tree(expr)}
    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(query))
    expected = {'data': [0, 0],
                'names': ['when_minute'],
                'datashape': '2 * int64'}
    assert result.status_code == RC.OK
    resp = serial.loads(result.data)
    assert list(serial.data_loads(resp['data'])) == expected['data']
    assert list(resp['names']) == expected['names']
    assert resp['datashape'] == expected['datashape']


@pytest.mark.parametrize('serial', all_formats)
def test_isin(test, serial):
    expr = t.events.value.isin(frozenset([1]))
    query = {'expr': to_tree(expr)}
    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(query))
    expected = {'data': [True, False],
                'names': ['value'],
                'datashape': '2 * bool'}
    assert result.status_code == RC.OK
    resp = serial.loads(result.data)
    assert list(serial.data_loads(resp['data'])) == expected['data']
    assert list(resp['names']) == expected['names']
    assert resp['datashape'] == expected['datashape']


@pytest.mark.parametrize('serial', all_formats)
def test_add_errors(temp_add_server, serial):
    pre_datashape = datashape.dshape(temp_add_server
                                     .get('/datashape')
                                     .data.decode('utf-8'))
    bunk_path = example('bunk.csv')
    blob = serial.dumps({'bunk': bunk_path})
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert response1.status_code == RC.UNPROCESSABLE_ENTITY

    # Test that the datashape of the server is accessible and unchanged after
    # trying to add a non-existent dataset.
    response2 = temp_add_server.get('/datashape')
    assert response2.status_code == RC.OK
    response_dshape = datashape.dshape(response2.data.decode('utf-8'))
    assert_dshape_equal(pre_datashape, response_dshape)


@pytest.mark.parametrize('serial', all_formats)
def test_add_default_not_allowed(temp_server, serial):
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': iris_path})
    response1 = temp_server.post('/add',
                                 headers=mimetype(serial),
                                 data=blob)
    assert 'NOT FOUND' in response1.status
    assert response1.status_code == RC.NOT_FOUND


@pytest.mark.parametrize('serial', all_formats)
def test_add_data_to_server(temp_add_server, serial):
    # add data
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': iris_path})
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert 'CREATED' in response1.status
    assert response1.status_code == RC.CREATED

    # check for expected server datashape
    response2 = temp_add_server.get('/datashape')
    expected2 = discover({'iris': data(iris_path)})
    response_dshape = datashape.dshape(response2.data.decode('utf-8'))
    assert_dshape_equal(response_dshape.measure.dict['iris'],
                        expected2.measure.dict['iris'])

    # compute on added data
    t = data({'iris': data(iris_path)})
    expr = t.iris.petal_length.sum()

    response3 = temp_add_server.post('/compute',
                                     data=serial.dumps({'expr': to_tree(expr)}),
                                     headers=mimetype(serial))

    result3 = serial.data_loads(serial.loads(response3.data)['data'])
    expected3 = compute(expr, {'iris': data(iris_path)})
    assert result3 == expected3


@pytest.mark.parametrize('serial', all_formats)
def test_cant_add_data_to_server(iris_server, serial):
    # try adding more data to server
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': iris_path})
    response1 = iris_server.post('/add',
                                 headers=mimetype(serial),
                                 data=blob)
    assert response1.status_code == RC.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize('serial', all_formats)
def test_add_data_twice_error(temp_add_server, serial):
    # add iris
    iris_path = example('iris.csv')
    payload = serial.dumps({'iris': iris_path})
    temp_add_server.post('/add',
                         headers=mimetype(serial),
                         data=payload)

    # Try to add to existing 'iris'
    resp = temp_add_server.post('/add',
                                headers=mimetype(serial),
                                data=payload)
    assert resp.status_code == RC.CONFLICT

    # Verify the server still serves the original 'iris'.
    response_ds = temp_add_server.get('/datashape').data.decode('utf-8')
    ds = datashape.dshape(response_ds)
    t = symbol('t', ds)
    query = {'expr': to_tree(t.iris)}
    resp = temp_add_server.post('/compute',
                                data=serial.dumps(query),
                                headers=mimetype(serial))
    assert resp.status_code == RC.OK


@pytest.mark.parametrize('serial', all_formats)
def test_add_two_data_sets_at_once_error(temp_add_server, serial):
    # Try to add two things at once
    payload = serial.dumps({'foo': 'iris.csv',
                            'bar': 'iris.csv'})
    resp = temp_add_server.post('/add',
                                headers=mimetype(serial),
                                data=payload)
    assert resp.status_code == RC.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize('serial', all_formats)
def test_add_bunk_data_error(temp_add_server, serial):
    # Try to add bunk data
    payload = serial.dumps({'foo': None})
    resp = temp_add_server.post('/add',
                                headers=mimetype(serial),
                                data=payload)
    assert resp.status_code == RC.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize('serial', all_formats)
def test_bad_add_payload(temp_add_server, serial):
    # try adding more data to server
    blob = serial.dumps('This is not a mutable mapping.')
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert response1.status_code == RC.UNPROCESSABLE_ENTITY


@pytest.mark.parametrize('serial', all_formats)
def test_add_expanded_payload(temp_add_server, serial):
    # Ensure that the expanded payload format is accepted by the server
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': {'source': iris_path,
                                  'kwargs': {'delimiter': ','}}})
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert 'CREATED' in response1.status
    assert response1.status_code == RC.CREATED


@pytest.mark.parametrize('serial', all_formats)
def test_add_expanded_payload_with_imports(temp_add_server, serial):
    # Ensure that the expanded payload format is accepted by the server
    iris_path = example('iris.csv')
    blob = serial.dumps({'iris': {'source': iris_path,
                                  'kwargs': {'delimiter': ','},
                                  'imports': ['csv']}})
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert 'CREATED' in response1.status
    assert response1.status_code == RC.CREATED


@pytest.mark.parametrize('serial', all_formats)
def test_add_expanded_payload_has_effect(temp_add_server, serial):
    # Ensure that the expanded payload format actually passes the arguments
    # through to the resource constructor
    iris_path = example('iris-latin1.tsv')
    csv_kwargs = {'delimiter': '\t', 'encoding': 'iso-8859-1'}
    blob = serial.dumps({'iris': {'source': iris_path,
                                  'kwargs': csv_kwargs}})
    response1 = temp_add_server.post('/add',
                                     headers=mimetype(serial),
                                     data=blob)
    assert 'CREATED' in response1.status
    assert response1.status_code == RC.CREATED

    # check for expected server datashape
    response2 = temp_add_server.get('/datashape')
    expected2 = discover({'iris': data(iris_path, **csv_kwargs)})
    response_dshape = datashape.dshape(response2.data.decode('utf-8'))
    assert_dshape_equal(response_dshape.measure.dict['iris'],
                        expected2.measure.dict['iris'])

    # compute on added data
    t = data({'iris': data(iris_path, **csv_kwargs)})
    expr = t.iris.petal_length.sum()

    response3 = temp_add_server.post('/compute',
                                     data=serial.dumps({'expr': to_tree(expr)}),
                                     headers=mimetype(serial))

    result3 = serial.data_loads(serial.loads(response3.data)['data'])
    expected3 = compute(expr, {'iris': data(iris_path, **csv_kwargs)})
    assert result3 == expected3


@pytest.mark.parametrize('serial', all_formats)
def test_odo_kwargs(test, serial):
    expr = t.dumb
    bad_query = {'expr': to_tree(expr)}

    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(bad_query))
    assert result.status_code == RC.INTERNAL_SERVER_ERROR
    assert b'return_df must be passed' in result.data

    good_query = {'expr': to_tree(expr),
                  'odo_kwargs': {'return_df': odo(DumbResource.df, list)}}
    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(good_query))
    assert result.status_code == RC.OK
    tdata = serial.loads(result.data)
    dshape = discover(DumbResource.df)
    assert_dshape_equal(datashape.dshape(tdata['datashape']),
                        dshape)
    assert_frame_equal(odo(serial.data_loads(tdata['data']),
                           DataFrame,
                           dshape=dshape),
                       DumbResource.df)


@pytest.mark.parametrize('serial', all_formats)
def test_compute_kwargs(test, serial):
    expr = t.dumb.sort()
    bad_query = {'expr': to_tree(expr)}

    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(bad_query))
    assert result.status_code == RC.INTERNAL_SERVER_ERROR
    assert b'return_df must be passed' in result.data

    good_query = {'expr': to_tree(expr),
                  'compute_kwargs': {'return_df': odo(DumbResource.df, list)}}
    result = test.post('/compute',
                       headers=mimetype(serial),
                       data=serial.dumps(good_query))
    assert result.status_code == RC.OK
    tdata = serial.loads(result.data)
    dshape = discover(DumbResource.df)
    assert_dshape_equal(datashape.dshape(tdata['datashape']),
                        dshape)
    assert_frame_equal(odo(serial.data_loads(tdata['data']),
                           DataFrame,
                           dshape=dshape),
                       DumbResource.df)


def test_fastmsgmpack_mutable_dataframe(test):
    expr = t.events  # just get back the dataframe
    query = {'expr': to_tree(expr)}
    result = test.post('/compute',
                       headers=mimetype(fastmsgpack),
                       data=fastmsgpack.dumps(query))
    assert result.status_code == RC.OK
    data = fastmsgpack.data_loads(fastmsgpack.loads(result.data)['data'])

    for block in data._data.blocks:
        # make sure all the blocks are mutable
        assert block.values.flags.writeable
