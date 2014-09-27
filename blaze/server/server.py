from __future__ import absolute_import, division, print_function


import blaze
from collections import Iterator
from flask import Flask, request, jsonify, json
from dynd import nd
from cytoolz import first
from functools import partial, wraps
from blaze import into, compute, compute_up
from ..api import discover, Table
from ..expr import Expr, TableSymbol, Selection, ColumnWise, TableSymbol
from ..expr import TableExpr
from ..expr.scalar.parser import exprify

from ..compatibility import map
from datashape import Mono

from .index import parse_index

# http://www.speedguide.net/port.php?port=6363
# http://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers
DEFAULT_PORT = 6363

class Server(object):
    """ Blaze Data Server

    Host local data through a web API

    >>> from pandas import DataFrame
    >>> df = DataFrame([[1, 'Alice',   100],
    ...                 [2, 'Bob',    -200],
    ...                 [3, 'Alice',   300],
    ...                 [4, 'Dennis',  400],
    ...                 [5,  'Bob',   -500]],
    ...                columns=['id', 'name', 'amount'])

    >>> server = Server({'accounts': df})
    >>> server.run() # doctest: +SKIP
    """
    __slots__ = 'app', 'datasets'

    def __init__(self, datasets=None):
        app = self.app = Flask('blaze.server.server')
        self.datasets = datasets or dict()

        for args, kwargs, func in routes:
            func2 = wraps(func)(partial(func, self.datasets))
            app.route(*args, **kwargs)(func2)

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        self.datasets[key] = value
        return value

    def run(self, *args, **kwargs):
        port = kwargs.pop('port', DEFAULT_PORT)
        return self.app.run(*args, port=port, **kwargs)


routes = list()

def route(*args, **kwargs):
    def f(func):
        routes.append((args, kwargs, func))
        return func
    return f


@route('/datasets.json')
def dataset(datasets):
    return jsonify(dict((k, str(discover(v))) for k, v in datasets.items()))


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
        rv = dset[index]
    except RuntimeError:
        return ("Bad index: %s" % (str(index)), 404)

    if isinstance(rv, Iterator):
        rv = list(rv)

    dshape = dset.dshape.subshape[index]
    rv = json.loads(str(nd.format_json(nd.array(rv, type=str(dshape)),
                                       tuple=True)))

    response = {'name': name,
                'index': data['index'],
                'datashape': str(dshape),
                'data': rv}

    return jsonify(response)


@route('/select/<name>.json', methods=['POST', 'PUT', 'GET'])
def select(datasets, name):
    """ Basic Selection API

    Allows remote querying of datasets.  Takes query data as JSON

    Takes requests like

    Example
    -------

    For the following array:

    [['Alice', 100],
     ['Bob', 200],
     ['Charlie', 300]]

    schema = '{name: string, amount: int32}'

    And the following

    url: /select/table-name.json
    POST-data: {'selection': 'amount >= 200',
                'columns': 'name'}

    and returns responses like

    {"name": "table-name",
     ...
     "datashape": "2 * string",
     "data": ["Bob", "Charlie"]}
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
    t = TableSymbol('t', dset.schema)
    dtypes = dict((c, t[c].dtype) for c in t.columns)

    columns = data.get('columns', None)
    if columns:
        try:
            columns = data['columns']
        except ValueError:
            return ("Bad columns", 404)
    try:
        select = exprify(data['selection'], dtypes)
    except (ValueError, KeyError):
        return ("Bad selection", 404)

    expr = Selection(t, ColumnWise(t, select))
    if columns:
        expr = expr[columns]
    try:
        rv = into([], compute(expr, dset))
    except RuntimeError:
        return ("Bad selection", 404)

    return jsonify({'name': name,
                    'columns': expr.columns,
                    'selection': str(select),
                    'datashape': str(expr.dshape),
                    'data': rv})



def to_tree(expr, names=None):
    """ Represent Blaze expression with core data structures

    Transform a Blaze expression into a form using only strings, dicts, lists
    and base types (int, float, datetime, ....)  This form can be useful for
    serialization.

    Parameters
    ----------

    expr: Blaze Expression

    Examples
    --------

    >>> t = TableSymbol('t', '{x: int32, y: int32}')
    >>> to_tree(t) # doctest: +SKIP
    {'op': 'TableSymbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}


    >>> to_tree(t.x.sum()) # doctest: +SKIP
    {'op': 'sum',
     'args': [
         {'op': 'Column',
         'args': [
             {
              'op': 'TableSymbol'
              'args': ['t', 'var * { x : int32, y : int32 }', False]
             }
             'x']
         }]
     }

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``TableSymbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Column', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t['x']

    See Also
    --------

    blaze.server.server.from_tree
    """
    if names and expr in names:
        return names[expr]
    if isinstance(expr, tuple):
        return [to_tree(arg, names=names) for arg in expr]
    elif isinstance(expr, Mono):
        return str(expr)
    elif isinstance(expr, Table):
        return to_tree(TableSymbol(expr._name, expr.schema), names)
    elif isinstance(expr, Expr):
        return {'op': type(expr).__name__,
                'args': [to_tree(arg, names) for arg in expr.args]}
    else:
        return expr

def expression_from_name(name):
    """

    >>> expression_from_name('By')
    <class 'blaze.expr.table.By'>
    """
    for signature, func in compute_up.funcs.items():
        try:
            if signature[0].__name__ == name:
                return signature[0]
        except TypeError:
            pass
    raise ValueError('%s not found in compute_up' % name)


def from_tree(expr, namespace=None):
    """ Convert core data structures to Blaze expression

    Core data structure representations created by ``to_tree`` are converted
    back into Blaze expressions.

    Parameters
    ----------

    expr: dict

    Examples
    --------

    >>> t = TableSymbol('t', '{x: int32, y: int32}')
    >>> tree = to_tree(t)
    >>> tree # doctest: +SKIP
    {'op': 'TableSymbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}

    >>> from_tree(tree)
    t

    >>> tree = to_tree(t.x.sum())
    >>> tree # doctest: +SKIP
    {'op': 'sum',
     'args': [
         {'op': 'Column',
         'args': [
             {
              'op': 'TableSymbol'
              'args': ['t', 'var * { x : int32, y : int32 }', False]
             }
             'x']
         }]
     }

    >>> from_tree(tree)
    sum(child=t['x'])

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``TableSymbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Column', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t['x']


    See Also
    --------

    blaze.server.server.to_tree
    """
    if isinstance(expr, dict):
        op, args = expr['op'], expr['args']
        if hasattr(blaze.expr, op):
            cls = getattr(blaze.expr, op)
        else:
            cls = expression_from_name(op)
        if 'Symbol' in op:
            children = [from_tree(arg) for arg in args]
        else:
            children = [from_tree(arg, namespace) for arg in args]
        return cls(*children)
    elif isinstance(expr, list):
        return tuple(from_tree(arg, namespace) for arg in expr)
    if namespace and expr in namespace:
        return namespace[expr]
    else:
        return expr


@route('/compute/<name>.json', methods=['POST', 'PUT', 'GET'])
def comp(datasets, name):
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

    t = TableSymbol(name, discover(dset))

    expr = from_tree(data['expr'], namespace={name: t})

    result = compute(expr, dset)
    if isinstance(expr, TableExpr):
        result = into(list, result)
    return jsonify({'name': name,
                    'datashape': str(expr.dshape),
                    'data': result})
