from __future__ import absolute_import, division, print_function


import blaze
from collections import Iterator
import socket
from flask import Flask, request, jsonify, json
from dynd import nd
from cytoolz import first, merge, valmap, assoc
from functools import partial, wraps
from blaze import into, compute
from .crossdomain import crossdomain
from blaze.expr import utils as expr_utils
from blaze.compute import compute_up
from datashape.predicates import iscollection
from ..api import discover, Data
from ..expr import Expr, Symbol, Selection, Broadcast, Symbol
from ..expr.parser import exprify
from .. import expr

from ..compatibility import map
from datashape import Mono

from .index import parse_index

__all__ = 'Server', 'to_tree', 'from_tree'

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
    __slots__ = 'app', 'datasets', 'port'

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
        self.port = port
        try:
            self.app.run(*args, port=port, **kwargs)
        except socket.error:
            print("\tOops, couldn't connect on port %d.  Is it busy?" % port)
            self.run(*args, **assoc(kwargs, 'port', port + 1))


routes = list()

def route(*args, **kwargs):
    def f(func):
        routes.append((args, kwargs, func))
        return func
    return f


@route('/datasets.json')
def dataset(datasets):
    return jsonify(dict((k, str(discover(v))) for k, v in datasets.items()))


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

    >>> t = Symbol('t', 'var * {x: int32, y: int32}')
    >>> to_tree(t) # doctest: +SKIP
    {'op': 'Symbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}


    >>> to_tree(t.x.sum()) # doctest: +SKIP
    {'op': 'sum',
     'args': [
         {'op': 'Column',
         'args': [
             {
              'op': 'Symbol'
              'args': ['t', 'var * { x : int32, y : int32 }', False]
             }
             'x']
         }]
     }

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``Symbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Column', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t.x

    See Also
    --------

    blaze.server.server.from_tree
    """
    if names and expr in names:
        return names[expr]
    if isinstance(expr, tuple):
        return [to_tree(arg, names=names) for arg in expr]
    if isinstance(expr, expr_utils._slice):
        return to_tree(expr.as_slice(), names=names)
    if isinstance(expr, slice):
        return {'op': 'slice',
                'args': [to_tree(arg, names=names) for arg in [expr.start, expr.stop, expr.step]]}
    elif isinstance(expr, Mono):
        return str(expr)
    elif isinstance(expr, Data):
        return to_tree(Symbol(expr._name, expr.dshape), names)
    elif isinstance(expr, Expr):
        return {'op': type(expr).__name__,
                'args': [to_tree(arg, names) for arg in expr._args]}
    else:
        return expr


def expression_from_name(name):
    """

    >>> expression_from_name('By')
    <class 'blaze.expr.split_apply_combine.By'>

    >>> expression_from_name('And')
    <class 'blaze.expr.arithmetic.And'>
    """
    import blaze
    if hasattr(blaze, name):
        return getattr(blaze, name)
    if hasattr(blaze.expr, name):
        return getattr(blaze.expr, name)
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

    >>> t = Symbol('t', 'var * {x: int32, y: int32}')
    >>> tree = to_tree(t)
    >>> tree # doctest: +SKIP
    {'op': 'Symbol',
     'args': ['t', 'var * { x : int32, y : int32 }', False]}

    >>> from_tree(tree)
    t

    >>> tree = to_tree(t.x.sum())
    >>> tree # doctest: +SKIP
    {'op': 'sum',
     'args': [
         {'op': 'Field',
         'args': [
             {
              'op': 'Symbol'
              'args': ['t', 'var * { x : int32, y : int32 }', False]
             }
             'x']
         }]
     }

    >>> from_tree(tree)
    sum(t.x)

    Simplify expresion using explicit ``names`` dictionary.  In the example
    below we replace the ``Symbol`` node with the string ``'t'``.

    >>> tree = to_tree(t.x, names={t: 't'})
    >>> tree # doctest: +SKIP
    {'op': 'Field', 'args': ['t', 'x']}

    >>> from_tree(tree, namespace={'t': t})
    t.x


    See Also
    --------

    blaze.server.server.to_tree
    """
    if isinstance(expr, dict):
        op, args = expr['op'], expr['args']
        if 'slice' == op:
            return slice(*[from_tree(arg, namespace) for arg in args])
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


@route('/compute/<name>.json', methods=['POST', 'PUT', 'GET', 'OPTIONS'])
@crossdomain(origin="*", headers=None)
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

    t = Symbol(name, discover(dset))
    namespace = data.get('namespace', dict())
    namespace[name] = t

    expr = from_tree(data['expr'], namespace=namespace)

    result = compute(expr, dset)
    if iscollection(expr.dshape):
        result = into(list, result)
    return jsonify({'name': name,
                    'datashape': str(expr.dshape),
                    'names' : t.columns,
                    'data': result})


@route('/ping')
def ping(datasets):
    return 'pong'

@route('/compute.json', methods=['POST', 'PUT', 'GET', 'OPTIONS'])
@crossdomain(origin="*", headers=None)
def compserver(datasets):
    if request.headers['content-type'] != 'application/json':
        return ("Expected JSON data", 404)
    try:
        data = json.loads(request.data)
    except ValueError:
        return ("Bad JSON.  Got %s " % request.data, 404)

    tree_ns = dict((name, Symbol(name, discover(datasets[name])))
                    for name in datasets)
    if 'namespace' in data:
        tree_ns = merge(tree_ns, data['namespace'])

    expr = from_tree(data['expr'], namespace=tree_ns)

    compute_ns = dict((Symbol(name, discover(datasets[name])), datasets[name])
                        for name in datasets)
    result = compute(expr, compute_ns)
    if iscollection(expr.dshape):
        result = into(list, result)

    return jsonify({'datashape': str(expr.dshape),
                    'data': result,
                    'names' : expr.fields,
                })
