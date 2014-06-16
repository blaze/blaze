from __future__ import absolute_import, division, print_function

from flask import Flask, request
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
        return "Expected JSON"
    try:
        data = json.loads(request.data)
    except ValueError:
        return "Bad JSON.  Got %s " % request.data

    try:
        dset = datasets[name]
    except KeyError:
        return "Dataset not found"

    try:
        index = parse_index(data['index'])
    except ValueError:
        return "Bad index"

    return json.dumps(dset.py.__getitem__(index))

if __name__ == '__main__':
    from blaze.data.python import Python

    accounts = Python([['Alice', 100], ['Bob', 200]],
                      schema='{name: string, amount: int32}')

    cities = Python([['Alice', 'NYC'], ['Bob', 'LA'], ['Charlie', 'Beijing']],
                      schema='{name: string, city: string}')

    datasets['accounts'] = accounts
    datasets['cities'] = cities
    app.run(debug=True)
