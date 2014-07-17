from __future__ import absolute_import, division, print_function

from dynd import nd

from blaze.serve.server import Server
from blaze.data.python import Python
from blaze.serve.index import parse_index, emit_index
from blaze.serve.client import Client

accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

server = Server(datasets={'accounts': accounts, 'cities': cities})

test = server.app.test_client()

import blaze.serve.client as client
client.requests = test # OMG monkey patching


dd = Client('http://localhost:5000', 'accounts')

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
