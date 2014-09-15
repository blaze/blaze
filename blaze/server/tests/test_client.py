from __future__ import absolute_import, division, print_function

from dynd import nd
from pandas import DataFrame

from blaze import TableSymbol, compute, Table, by, into, TableExpr
from blaze.dispatch import dispatch
from blaze.server import Server
from blaze.data.python import Python
from blaze.server.index import parse_index, emit_index
from blaze.server.client import Client, ExprClient, discover, resource

accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

df = DataFrame([['Alice', 100], ['Bob', 200]],
               columns=['name', 'amount'])

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

server = Server(datasets={'accounts': accounts,
                          'accounts_df': df,
                          'cities': cities})

test = server.app.test_client()

from blaze.server import client
client.requests = test # OMG monkey patching


dd = Client('http://localhost:6363', 'accounts')

def test_dshape():
    assert dd.dshape == accounts.dshape


def test_get_py():
    assert list(dd[0:, 'name']) == list(accounts[:, 'name'])


def test_get_dynd():
    result = dd.dynd[0:, 'name']
    expected = accounts.dynd[:, 'name']
    assert nd.as_py(result) == nd.as_py(expected)


def test_iter():
    assert list(dd) == list(accounts)


def test_chunks():
    assert [nd.as_py(chunk) for chunk in dd.chunks()] == \
            [nd.as_py(chunk) for chunk in accounts.chunks()]


def test_expr_client():
    ec = ExprClient('localhost:6363', 'accounts_df')
    assert discover(ec) == discover(df)

    t = TableSymbol('t', discover(ec))
    expr = t.amount.sum()

    assert compute(expr, ec) == 300
    assert compute(t.name, ec) == ['Alice', 'Bob']


def test_expr_client_interactive():
    ec = ExprClient('localhost:6363', 'accounts_df')
    t = Table(ec)

    assert compute(t.name) == ['Alice', 'Bob']
    assert (into(set, compute(by(t.name, min=t.amount.min(),
                                         max=t.amount.max()))) ==
            set([('Alice', 100, 100), ('Bob', 200, 200)]))


def test_resource():
    ec = resource('blaze://localhost:6363', 'accounts_df')
    assert discover(ec) == discover(df)

def test_resource_default_port():
    ec = resource('blaze://localhost', 'accounts_df')
    assert discover(ec) == discover(df)

def test_resource_all_in_one():
    ec = resource('blaze://localhost:6363::accounts_df')
    assert discover(ec) == discover(df)


class CustomExpr(TableExpr):
    __slots__ = 'child',

    @property
    def dshape(self):
        return self.child.dshape


@dispatch(CustomExpr, DataFrame)
def compute_one(expr, data, **kwargs):
    return data


def test_custom_expressions():
    ec = ExprClient('localhost:6363', 'accounts_df')
    t = TableSymbol('t', discover(ec))

    assert list(map(tuple, compute(CustomExpr(t), ec))) == into(list, df)
