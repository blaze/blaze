from __future__ import absolute_import, division, print_function

import pytest
pytest.importorskip('flask')

from pandas import DataFrame
from blaze import compute, Data, by, into
from blaze.expr import Expr, Symbol, Field
from blaze.dispatch import dispatch
from blaze.server import Server
from blaze.server.index import parse_index, emit_index
from blaze.server.client import Client, discover, resource, ClientDataset

df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])
df2 = DataFrame([['Charlie', 100], ['Dan', 200]],
                columns=['name', 'amount'])

server = Server(datasets={'accounts': df, 'accounts2': df})

test = server.app.test_client()

from blaze.server import client
client.requests = test # OMG monkey patching


def test_client():
    c = Client('localhost:6363')
    assert str(discover(c)) == str(discover(server.datasets))

    t = Symbol('t', discover(c))
    expr = t.accounts.amount.sum()

    assert compute(expr, c) == 300
    assert 'name' in t.accounts.fields
    assert isinstance(t.accounts.name, Field)
    assert compute(t.accounts.name, c) == ['Alice', 'Bob']


def test_compute_with_dataset():
    c = resource('blaze://localhost:6363::accounts')
    s = Symbol('s', discover(c))

    assert compute(s.name, c) == ['Alice', 'Bob']

def test_expr_client_interactive():
    c = Client('localhost:6363')
    t = Data(c)

    assert compute(t.accounts.name) == ['Alice', 'Bob']
    assert (into(set, compute(by(t.accounts.name, min=t.accounts.amount.min(),
                                                  max=t.accounts.amount.max())))
            == set([('Alice', 100, 100), ('Bob', 200, 200)]))

def test_compute_client_with_multiple_datasets():
    c = resource('blaze://localhost:6363')
    s = Symbol('s', discover(c))

    assert compute(s.accounts.amount.sum() + s.accounts2.amount.sum(),
                    {s: c}) == 600

def test_compute_multiple_client_datasets():
    a1 = resource('blaze://localhost:6363::accounts')
    a2 = resource('blaze://localhost:6363::accounts2')
    s1 = Symbol('s1', discover(a1))
    s2 = Symbol('s2', discover(a2))

    assert compute(s1.amount.sum() + s2.amount.sum(),
            {s1: a1, s2: a2}) == 600

def test_clientdataset_into_list():
    a = resource('blaze://localhost:6363::accounts')

    assert into(list, a) == [['Alice', 100], ['Bob', 200]]


def test_clientdataset_into_list():
    a = resource('blaze://localhost:6363::accounts')

    assert str(into(DataFrame, a)) == \
            str(DataFrame([['Alice', 100], ['Bob', 200]],
                          columns=['name', 'amount']))

def test_resource():
    c = resource('blaze://localhost:6363')
    assert str(discover(c)) == str(discover(server.datasets))


def test_resource_with_dataset():
    c = resource('blaze://localhost:6363::accounts')
    assert str(discover(c)) == str(discover(df))


def test_resource_default_port():
    ec = resource('blaze://localhost')
    assert str(discover(ec)) == str(discover(server.datasets))


def test_resource_non_default_port():
    ec = resource('blaze://localhost:6364')
    assert ec.url == 'http://localhost:6364'


def test_resource_all_in_one():
    ec = resource('blaze://localhost:6363')
    assert str(discover(ec)) == str(discover(server.datasets))


class CustomExpr(Expr):
    __slots__ = '_hash', '_child'

    @property
    def dshape(self):
        return self._child.dshape


@dispatch(CustomExpr, DataFrame)
def compute_up(expr, data, **kwargs):
    return data


def test_custom_expressions():
    ec = Client('localhost:6363')
    t = Symbol('t', discover(ec))

    assert list(map(tuple, compute(CustomExpr(t.accounts), ec))) == into(list, df)
