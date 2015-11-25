from __future__ import absolute_import, division, print_function

import warnings

from datashape import dshape
import flask
from flask.testing import FlaskClient
from odo import resource
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from ..expr import Expr
from ..dispatch import dispatch
from .server import DEFAULT_PORT
from .serialization import json

# These are a hack for testing
# It's convenient to use requests for live production but use
# flask for testing.  Sadly they have different Response objects,
# hence the dispatched functions

__all__ = 'Client',


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


def _request(method, client, url, params=None, auth=None, **kwargs):
    if not isinstance(requests, FlaskClient):
        kwargs['verify'] = client.verify_ssl
        kwargs['params'] = params
        kwargs['auth'] = auth

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InsecureRequestWarning)
        return method(
            '{base}{url}'.format(
                base=client.url,
                url=url,
            ),
            **kwargs
        )


def get(*args, **kwargs):
    return _request(requests.get, *args, **kwargs)


def post(*args, **kwargs):
    return _request(requests.post, *args, **kwargs)


class Client(object):

    """ Client for Blaze Server

    Provides programmatic access to datasets living on Blaze Server

    Parameters
    ----------

    url : str
        URL of a Blaze server
    serial : SerializationFormat, optional
        The serialization format object to use. Defaults to JSON.
        A serialization format is an object that supports:
        name, loads, and dumps.
    verify_ssl : bool, optional
        Verify the ssl certificate from the server.
        This is enabled by default.
    auth : tuple, optional
        The username and password to use when connecting to the server.
        If not provided, no auth header will be sent.
    chunksize : int, optional
        The size of the chunks to use on the server when making queries.
        If this is None, then no chunking will be used.

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
    __slots__ = 'url', 'serial', 'verify_ssl', 'auth', 'chunksize'

    def __init__(self, url, serial=json,
                 verify_ssl=True, auth=None, chunksize=None):
        if chunksize and serial.stream_unpacker is None:
            raise TypeError(
                'serial must support stream_unpacker if chunksize > 0',
            )

        url = url.strip('/')
        if not url.startswith('http'):
            url = 'http://' + url
        self.url = url
        self.serial = serial
        self.verify_ssl = verify_ssl
        self.auth = auth
        self.chunksize = chunksize

    @property
    def dshape(self):
        """The datashape of the client"""
        response = get(self, '/datashape', auth=self.auth)
        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        return dshape(content(response).decode('utf-8'))


@dispatch(Client)
def discover(c):
    return c.dshape


def mimetype(serial):
    """Function to generate a blaze serialization format mimetype put into a
    dictionary of headers for consumption by requests.

    Examples
    --------
    >>> from blaze.server.serialization import msgpack
    >>> mimetype(msgpack)
    {'Content-Type': 'application/vnd.blaze+msgpack'}
    """
    return {'Content-Type': 'application/vnd.blaze+%s' % serial.name}


def read_data(raw_data, serial, chunking, extract=False):
    """Read raw data using the serialization format and chunking scheme.

    Parameters
    ----------
    raw_data : bytes
        The raw data to read.
    serial : SerializationFormat
        The format to use to decode the data.
    chunking : bool
        Decode the data as chunks.
    extract : bool, optional
        Extract the data field out of non-chunking results.
    """
    if chunking:
        unpacker = serial.stream_unpacker()
        unpacker.feed(raw_data)
        obj = unpacker.unpack()
        if isinstance(obj, list):
            obj = obj + sum(unpacker, [])
        else:
            obj = obj['d']
        return obj

    obj = serial.loads(raw_data)
    if extract:
        obj = obj['data']
    return obj


@dispatch(Expr, Client)
def compute_down(expr, ec, **kwargs):
    from .server import to_tree
    tree = to_tree(expr)

    serial = ec.serial
    chunksize = ec.chunksize
    if chunksize is not None:
        params = {'chunksize': ec.chunksize}
        endpoint = '/compute/stream'
    else:
        params = {}
        endpoint = '/compute'

    r = post(
        ec,
        endpoint,
        data=serial.dumps({'expr': tree}),
        auth=ec.auth,
        headers=mimetype(serial),
        params=params
    )

    if not ok(r):
        raise ValueError("Bad response: %s" % reason(r))
    return read_data(content(r), serial, chunksize, extract=True)


@resource.register('blaze://.+')
def resource_blaze(uri, leaf=None, **kwargs):
    if leaf is not None:
        raise ValueError('The syntax blaze://...::{leaf} is no longer '
                         'supported as of version 0.8.1.\n'
                         'You can access {leaf!r} using this syntax:\n'
                         'Data({uri})[{leaf!r}]'
                         .format(leaf=leaf, uri=uri))
    uri = uri[len('blaze://'):]
    sp = uri.split('/')
    tld, rest = sp[0], sp[1:]
    if ':' not in tld:
        tld += ':%d' % DEFAULT_PORT
    uri = '/'.join([tld] + list(rest))
    return Client(uri)
