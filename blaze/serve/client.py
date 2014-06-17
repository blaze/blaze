from ..data.core import DataDescriptor
from ..data.utils import coerce
from .index import emit_index
import requests
import json
import flask
from datashape import dshape

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

    def get_py(self, key):
        print(json.dumps(emit_index(key)))
        response = requests.put('%s/data/%s.json' % (self.url, self.name),
                                data=json.dumps({'index': emit_index(key)}),
                                headers = {'Content-type': 'application/json',
                                           'Accept': 'text/plain'})
        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        data = json.loads(content(response))

        print(data['datashape'])
        print(data['data'])
        return coerce(data['datashape'], data['data'])

    @property
    def dshape(self):
        response = requests.get('%s/datasets.json' % self.url)

        if not ok(response):
            raise ValueError("Bad Response: %s" % reason(response))

        data = json.loads(content(response))

        return dshape(data[self.name])
