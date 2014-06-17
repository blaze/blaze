from __future__ import absolute_import, division, print_function

from collections import Iterator
from flask import Flask, request, jsonify, json
from functools import partial, wraps

from .index import parse_index

class Server(object):
    __slots__ = 'app', 'datasets'

    def __init__(self, name='Blaze-Server', datasets=None):
        app = self.app = Flask(name)
        self.datasets = datasets or dict()

        for args, kwargs, func in routes:
            func2 = wraps(func)(partial(func, self.datasets))
            app.route(*args, **kwargs)(func2)

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        self.datasets[key] = value
        return value


routes = list()

def route(*args, **kwargs):
    def f(func):
        routes.append((args, kwargs, func))
        return func
    return f


@route('/datasets.json')
def dataset(datasets):
    return jsonify(dict((k, str(v.dshape)) for k, v in datasets.items()))


@route('/data/<name>.json', methods=['POST', 'PUT', 'GET'])
def data(datasets, name):
    """ Basic indexing API

    Allows remote indexing of datasets.  Takes indexing data as JSON

    Takes requests like
    Example
    -------

    For the following array:

    [['Alice', 100],
     ['Bob', 200],
     ['Charlie', 300]]

    schema = '{name: string, amount: int32}'

    And the following

    url: /data/table-name.json
    POST-data: {'index': [{'start': 0, 'step': 3}, 'name']}

    and returns responses like

    {"name": "table-name",
     "index": [0, "name"],
     "datashape": "3 * string",
     "data": ["Alice", "Bob", "Charlie"]}
     """

    if request.headers['content-type'] != 'application/json':
        return ("Expected JSON data", 404)
    try:
        data = json.loads(request.data)
    except ValueError:
        return ("Bad JSON.  Got %s " % request.data, 404)

    try:
        dset = datasets[name]
    except KeyError:
        return ("Dataset %s not found" % name, 404)

    try:
        index = parse_index(data['index'])
    except ValueError:
        return ("Bad index", 404)

    try:
        rv = dset.py[index]
    except RuntimeError:
        return ("Bad index: %s" % (str(index)), 404)

    if isinstance(rv, Iterator):
        rv = list(rv)

    return jsonify({'name': name,
                    'index': data['index'],
                    'datashape': str(dset.dshape.subshape[index]),
                    'data': rv})
