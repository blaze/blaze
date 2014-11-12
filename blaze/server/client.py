from __future__ import absolute_import, division, print_function

import requests
from flask import json
import flask
from dynd import nd
from datashape import dshape, DataShape, Record

from ..data import DataDescriptor
from ..data.utils import coerce
from ..expr import Expr, Symbol
from ..dispatch import dispatch
from .index import emit_index
from ..resource import resource
from .server import DEFAULT_PORT

# These are a hack for testing
# It's convenient to use requests for live production but use
# flask for testing.  Sadly they have different Response objects,
# hence the dispatched functions

__all__ = 'Client', 'ExprClient'

def content(response):
    if isinstance(response, flask.Response):
        return response.data
    if isinstance(response, requests.Response):
        return response.content

def ok(response):
    if isinstance(response, flask.Response):
        return 'OK' in response.status
    if isinstance(response, requests.Response):
        return response.ok

def reason(response):
    if isinstance(response, flask.Response):
        return response.status
    if isinstance(response, requests.Response):
        return response.reason


class Client(object):
    """ Client for Blaze Server

    Provides programmatic access to datasets living on Blaze Server

    Parameters
    ----------

    url: str
        URL of a Blaze server

    Examples
    --------

    >>> # This example matches with the docstring of ``Server``
    >>> c = Client('localhost:6363')
    >>> t = Data(c) # doctest: +SKIP

    See Also
    --------

    blaze.server.server.Server
    """
    __slots__ = 'url'
    def __init__(self, url, **kwargs):
        url = url.strip('/')
        if not url[:4] == 'http':
            url = 'http://' + url
        self.url = url

    @property
    def dshape(self):
        response = requests.get('%s/datasets.json' % self.url)

        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        data = json.loads(content(response))

        return DataShape(Record([[name, dshape(ds)] for name, ds in
            data.items()]))


class ClientDataset(object):
    """ A dataset residing on a foreign Blaze Server

    Not for public use.  Suggest the use of ``blaze.server.client.Client``
    class instead.

    This is only used to support backwards compatibility for the syntax

        Data('blaze://hostname::dataname')

    The following behavior is suggested instead

        Data('blaze://hostname').dataname
    """
    __slots__ = 'client', 'name'
    def __init__(self, client, name):
        self.client = client
        self.name = name

    @property
    def dshape(self):
        return self.client.dshape.measure.dict[self.name]


def ExprClient(*args, **kwargs):
    import warnings
    warnings.warn("Deprecated use `Client` instead", DeprecationWarning)
    return Client(*args, **kwargs)


@dispatch((Client, ClientDataset))
def discover(ec):
    return ec.dshape


@dispatch(Expr, ClientDataset)
def compute_down(expr, data, **kwargs):
    s = Symbol('client', discover(data.client))
    leaf = expr._leaves()[0]
    return compute_down(expr._subs({leaf: s[data.name]}), data.client, **kwargs)

@dispatch(Expr, Client)
def compute_down(expr, ec, **kwargs):
    from .server import to_tree
    from ..api import Data
    from ..api import into
    from pandas import DataFrame
    leaf = expr._leaves()[0]
    tree = to_tree(expr, dict((leaf[f], f) for f in leaf.fields))

    r = requests.get('%s/compute.json' % ec.url,
                     data = json.dumps({'expr': tree}),
                     headers={'Content-Type': 'application/json'})

    if not ok(r):
        raise ValueError("Bad response: %s" % reason(r))

    data = json.loads(content(r))

    return data['data']

@resource.register('blaze://.+::\w+', priority=16)
def resource_blaze_dataset(uri, **kwargs):
    uri, name = uri.split('::')
    client = resource(uri)
    return ClientDataset(client, name)


@resource.register('blaze://.+')
def resource_blaze(uri, **kwargs):
    uri = uri[len('blaze://'):]
    sp = uri.split('/')
    tld, rest = sp[0], sp[1:]
    if ':' not in tld:
        tld = tld + ':%d' % DEFAULT_PORT
    uri = '/'.join([tld] + list(rest))
    return Client(uri)
