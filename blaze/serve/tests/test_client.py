from blaze.serve.server import Server
from blaze.data.python import Python
from blaze.serve.index import parse_index, emit_index
from blaze.serve.client import Client

from dynd import nd


accounts = Python([['Alice', 100], ['Bob', 200]],
                  schema='{name: string, amount: int32}')

cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                  schema='{name: string, city: string}')

server = Server(datasets={'accounts': accounts, 'cities': cities})

test = server.app.test_client()

import blaze.serve.client as client
client.requests = test # OMG monkey patching



def test_dshape():
    dd = Client('http://localhost:5000', 'accounts')
    assert dd.dshape == accounts.dshape


def test_get_py():
    dd = Client('http://localhost:5000', 'accounts')
    assert list(dd.py[0:, 'name']) == list(accounts.py[:, 'name'])

def test_get_dynd():
    dd = Client('http://localhost:5000', 'accounts')
    result = dd.dynd[0:, 'name']
    expected = accounts.dynd[:, 'name']
    assert nd.as_py(result) == nd.as_py(expected)

