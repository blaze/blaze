from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')

from pandas import DataFrame
from blaze import compute, Data, by, into, discover
from blaze.expr import Expr, symbol, Field
from blaze.dispatch import dispatch
from blaze.server import Server
from blaze.server.client import Client, resource

from blaze.server.tests.fixtures import make_app_context_fixture


df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])

df2 = DataFrame([['Charlie', 100], ['Dan', 200]],
                columns=['name', 'amount'])

data = {'accounts': df, 'accounts2': df}

server = Server(data)

test = server.app.test_client()

from blaze.server import client
client.requests = test # OMG monkey patching


app_context = make_app_context_fixture(server)


def test_client(app_context):
    c = Client('localhost:6363')
    assert str(discover(c)) == str(discover(data))

    t = symbol('t', discover(c))
    expr = t.accounts.amount.sum()

    assert compute(expr, c) == 300
    assert 'name' in t.accounts.fields
    assert isinstance(t.accounts.name, Field)
    assert compute(t.accounts.name, c) == ['Alice', 'Bob']


def test_expr_client_interactive(app_context):
    c = Client('localhost:6363')
    t = Data(c)

    assert compute(t.accounts.name) == ['Alice', 'Bob']
    assert (into(set, compute(by(t.accounts.name, min=t.accounts.amount.min(),
                                                  max=t.accounts.amount.max())))
            == set([('Alice', 100, 100), ('Bob', 200, 200)]))


def test_compute_client_with_multiple_datasets(app_context):
    c = resource('blaze://localhost:6363')
    s = symbol('s', discover(c))

    assert compute(s.accounts.amount.sum() + s.accounts2.amount.sum(),
                    {s: c}) == 600


def test_resource(app_context):
    c = resource('blaze://localhost:6363')
    assert isinstance(c, Client)
    assert str(discover(c)) == str(discover(data))


def test_resource_default_port(app_context):
    ec = resource('blaze://localhost')
    assert str(discover(ec)) == str(discover(data))


def test_resource_non_default_port(app_context):
    ec = resource('blaze://localhost:6364')
    assert ec.url == 'http://localhost:6364'


def test_resource_all_in_one(app_context):
    ec = resource('blaze://localhost:6363')
    assert str(discover(ec)) == str(discover(data))


class CustomExpr(Expr):
    __slots__ = '_hash', '_child'

    @property
    def dshape(self):
        return self._child.dshape


@dispatch(CustomExpr, DataFrame)
def compute_up(expr, data, **kwargs):
    return data


def test_custom_expressions(app_context):
    ec = Client('localhost:6363')
    t = symbol('t', discover(ec))

    assert list(map(tuple, compute(CustomExpr(t.accounts), ec))) == into(list, df)


def test_client_dataset(app_context):
    d = Data('blaze://localhost::accounts')
    assert list(map(tuple, into(list, d))) == into(list, df)
