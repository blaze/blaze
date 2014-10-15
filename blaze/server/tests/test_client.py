from __future__ import absolute_import, division, print_function

from dynd import nd
from pandas import DataFrame

from blaze import Expr, TableSymbol, compute, Table, by, into, Field
from blaze.dispatch import dispatch
from blaze.server import Server
from blaze.server.index import parse_index, emit_index
from blaze.server.client import Client, discover, resource

df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])

server = Server(datasets={'accounts': df})

test = server.app.test_client()

from blaze.server import client
client.requests = test # OMG monkey patching


def test_expr_client():
    ec = Client('localhost:6363', 'accounts')
    assert discover(ec) == discover(df)

    t = TableSymbol('t', discover(ec))
    expr = t.amount.sum()

    assert compute(expr, ec) == 300
    assert 'name' in t.fields
    assert isinstance(t.name, Field)
    assert compute(t.name, ec) == ['Alice', 'Bob']


def test_expr_client_interactive():
    ec = Client('localhost:6363', 'accounts')
    t = Table(ec)

    assert compute(t.name) == ['Alice', 'Bob']
    assert (into(set, compute(by(t.name, min=t.amount.min(),
                                         max=t.amount.max()))) ==
            set([('Alice', 100, 100), ('Bob', 200, 200)]))


def test_resource():
    ec = resource('blaze://localhost:6363', 'accounts')
    assert discover(ec) == discover(df)

def test_resource_default_port():
    ec = resource('blaze://localhost', 'accounts')
    assert discover(ec) == discover(df)

def test_resource_all_in_one():
    ec = resource('blaze://localhost:6363::accounts')
    assert discover(ec) == discover(df)


class CustomExpr(Expr):
    __slots__ = '_child',

    @property
    def dshape(self):
        return self._child.dshape


@dispatch(CustomExpr, DataFrame)
def compute_up(expr, data, **kwargs):
    return data


def test_custom_expressions():
    ec = Client('localhost:6363', 'accounts')
    t = TableSymbol('t', discover(ec))

    assert list(map(tuple, compute(CustomExpr(t), ec))) == into(list, df)
