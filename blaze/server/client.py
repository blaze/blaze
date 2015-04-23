from __future__ import absolute_import, division, print_function

try:
    import flask
    from flask import json
    import requests
except ImportError:
    pass


from odo import resource
from datashape import dshape

from ..expr import Expr
from ..dispatch import dispatch
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
        return response.text


class Client(object):

    """ Client for Blaze Server

    Provides programmatic access to datasets living on Blaze Server

    Parameters
    ----------

    url : str
        URL of a Blaze server

    Examples
    --------

    >>> # This example matches with the docstring of ``Server``
    >>> from blaze import Data
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
        """The datashape of the client"""
        response = requests.get('%s/datashape' % self.url)

        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        return dshape(content(response).decode('utf-8'))


def ExprClient(*args, **kwargs):
    import warnings
    warnings.warn("Deprecated use `Client` instead", DeprecationWarning)
    return Client(*args, **kwargs)


@dispatch(Client)
def discover(c):
    return c.dshape


@dispatch(Expr, Client)
def compute_down(expr, ec, **kwargs):
    from .server import to_tree
    tree = to_tree(expr)

    r = requests.get('%s/compute.json' % ec.url,
                     data=json.dumps({'expr': tree}),
                     headers={'Content-Type': 'application/json'})

    if not ok(r):
        raise ValueError("Bad response: %s" % reason(r))

    data = json.loads(content(r).decode('utf-8'))

    return data['data']


@resource.register('blaze://.+')
def resource_blaze(uri, **kwargs):
    uri = uri[len('blaze://'):]
    sp = uri.split('/')
    tld, rest = sp[0], sp[1:]
    if ':' not in tld:
        tld = tld + ':%d' % DEFAULT_PORT
    uri = '/'.join([tld] + list(rest))
    return Client(uri)
