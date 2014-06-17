from __future__ import absolute_import, division, print_function

import requests
from flask import json
import flask
from dynd import nd
from datashape import dshape

from ..data.core import DataDescriptor
from ..data.utils import coerce
from .index import emit_index

# These are a hack for testing
# It's convenient to use requests for live production but use
# flask for testing.  Sadly they have different Response objects,
# hence the dispatched functions

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


class Client(DataDescriptor):
    __slots__ = 'uri', 'name'
    def __init__(self, url, name):
        self.url = url.strip('/')
        self.name = name

    def _get_data(self, key):
        response = requests.put('%s/data/%s.json' % (self.url, self.name),
                                data=json.dumps({'index': emit_index(key)}),
                                headers = {'Content-type': 'application/json',
                                           'Accept': 'text/plain'})
        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        data = json.loads(content(response))

        return data['datashape'], data['data']

    def get_py(self, key):
        dshape, data = self._get_data(key)
        return coerce(dshape, data)

    def get_dynd(self, key):
        dshape, data = self._get_data(key)
        return nd.array(data, type=str(dshape))


    @property
    def dshape(self):
        response = requests.get('%s/datasets.json' % self.url)

        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        data = json.loads(content(response))

        return dshape(data[self.name])
