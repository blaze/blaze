from __future__ import absolute_import, division, print_function

from collections import Iterator
from flask import Flask, request, jsonify
import json

from .index import parse_index

app = Flask('Blaze-Server')

datasets = dict()

@app.route('/')
def hello():
    return 'Welcome to Blaze Server'

@app.route('/datasets.json')
def dataset():
    return '\n'.join(': '.join((name, str(dset.dshape))) for name, dset in
                                datasets.items())


@app.route('/data/<name>.json', methods=['POST'])
def data(name):
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

    rv = dset.py.__getitem__(index)
    if isinstance(rv, Iterator):
        rv = list(rv)

    return jsonify({'index': data['index'],
                    'datashape': str(dset.dshape.subshape.__getitem__(index)),
                    'data': rv})
