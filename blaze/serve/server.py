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
    return jsonify(dict((k, str(v.dshape)) for k, v in datasets.items()))


@app.route('/data/<name>.json', methods=['POST'])
def data(name):
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

    rv = dset.py.__getitem__(index)
    if isinstance(rv, Iterator):
        rv = list(rv)

    return jsonify({'name': name,
                    'index': data['index'],
                    'datashape': str(dset.dshape.subshape.__getitem__(index)),
                    'data': rv})
