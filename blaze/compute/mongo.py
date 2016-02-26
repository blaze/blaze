""" MongoDB Backend - Uses aggregation pipeline

If you don't have a mongo server running

    $ conda install mongodb -y
    $ mongod &

>> from blaze import *

>> data = [(1, 'Alice', 100),
...        (2, 'Bob', -200),
...        (3, 'Charlie', 300),
...        (4, 'Denis', 400),
...        (5, 'Edith', -500)]

Migrate data into MongoDB

>> import pymongo
>> db = pymongo.MongoClient().db
>> db.mydata.drop()  # clear out old results
>> _ = into(db.mydata, data, columns=['id', 'name', 'amount'])

Objective: find the name of accounts with negative amount

Using MongoDB query language

>> db.mydata.aggregate([{'$match': {'amount': {'$lt': 0}}}, # doctest: +SKIP
..                      {'$project': {'name': 1, '_id': 0}}])['result']
[{'name': 'Bob'}, {'name': 'Edith'}]

Using Blaze

>> t = data(db.mydata)
>> t[t.amount < 0].name
    name
0    Bob
1  Edith

>> db.mydata.drop()

Uses the aggregation pipeline
http://docs.mongodb.org/manual/core/aggregation-pipeline/
"""

from __future__ import absolute_import, division, print_function

import numbers
import fnmatch

from operator import attrgetter

from pymongo.collection import Collection
from pymongo.database import Database

from datashape.predicates import isscalar
from toolz import pluck, first, get, compose
import toolz
import datetime

from ..expr import (
    And,
    Arithmetic,
    Broadcast,
    By,
    Distinct,
    Eq,
    Expr,
    Field,
    FloorDiv,
    Ge,
    Gt,
    Head,
    Le,
    Lt,
    Ne,
    Or,
    Projection,
    Reduction,
    SimpleSelection,
    Sort,
    Summary,
    Like,
    Symbol,
    ceil,
    count,
    floor,
    math,
    max,
    mean,
    min,
    nelements,
    nunique,
    sum,
    symbol,
)
from ..expr.broadcast import broadcast_collect, Broadcastable
from ..expr.datetime import Day, Month, Year, Minute, Second, UTCFromTimestamp
from ..expr.optimize import simple_selections
from ..compatibility import _strtypes
from ..dispatch import dispatch


__all__ = ['MongoQuery']


@dispatch(Expr, Collection)
def pre_compute(expr, data, scope=None, **kwargs):
    return MongoQuery(data, [])


@dispatch(Expr, Database)
def pre_compute(expr, data, **kwargs):
    return data


class MongoQuery(object):
    """
    A Pair of a pymongo collection and a aggregate query

    We need to carry around both a pymongo collection and a list of
    dictionaries to feed into the aggregation pipeline.  This class
    carries around such a pair.

    Parameters
    ----------
    coll: pymongo.collection.Collection
        A single pymongo collection
    query: list of dicts
        A query to send to coll.aggregate

    >>> q = MongoQuery(db.my_collection, # doctest: +SKIP
    ...     [{'$match': {'name': 'Alice'}},
    ...      {'$project': {'name': 1, 'amount': 1, '_id': 0}}])
    """
    def __init__(self, coll, query):
        self.coll = coll
        self.query = tuple(query)

    def append(self, clause):
        return MongoQuery(self.coll, self.query + (clause,))

    def info(self):
        return self.coll, self.query

    def __eq__(self, other):
        return type(self) == type(other) and self.info() == other.info()

    def __hash__(self):
        return hash((type(self), self.info()))


@dispatch(Expr, (MongoQuery, Collection))
def optimize(expr, seq):
    return broadcast_collect(simple_selections(expr), no_recurse=SimpleSelection)


@dispatch(Head, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$limit': t.n})


@dispatch(Broadcast, MongoQuery)
def compute_up(t, q, **kwargs):
    s = t._scalars[0]
    d = dict((s[c], symbol(c, s[c].dshape.measure)) for c in s.fields)
    expr = t._scalar_expr._subs(d)
    name = expr._name or 'expr_%d' % abs(hash(expr))
    return q.append({'$project': {name: compute_sub(expr)}})


binops = {'+': 'add',
          '*': 'multiply',
          '/': 'divide',
          '-': 'subtract',
          '%': 'mod'}


def compute_sub(t):
    """
    Build an expression tree in a MongoDB compatible way.

    Parameters
    ----------
    t : Arithmetic
        Scalar arithmetic expression

    Returns
    -------
    sub : dict
        An expression tree

    Examples
    --------
    >>> from blaze import Symbol
    >>> s = symbol('s', 'float64')
    >>> expr = s * 2 + s - 1
    >>> expr
    ((s * 2) + s) - 1
    >>> compute_sub(expr)
    {'$subtract': [{'$add': [{'$multiply': ['$s', 2]}, '$s']}, 1]}

    >>> when = symbol('when', 'datetime')
    >>> compute_sub(s + when.day)
    {'$add': ['$s', {'$dayOfMonth': '$when'}]}
    """
    if isinstance(t, _strtypes + (Symbol,)):
        return '$%s' % t
    elif isinstance(t, numbers.Number):
        return t
    elif isinstance(t, FloorDiv):
        return compute_sub(floor(t.lhs / t.rhs))
    elif isinstance(t, Arithmetic) and hasattr(t, 'symbol') and t.symbol in binops:
        op = binops[t.symbol]
        return {'$%s' % op: [compute_sub(t.lhs), compute_sub(t.rhs)]}
    elif isinstance(t, floor):
        x = compute_sub(t._child)
        return {'$subtract': [x, {'$mod': [x, 1]}]}
    elif isinstance(t, math.abs):
        x = compute_sub(t._child)
        return {'$cond': [{'$lt': [x, 0]},
                          {'$subtract': [0, x]},
                          x]}
    elif isinstance(t, ceil):
        x = compute_sub(t._child)
        return {'$add': [x, {'$subtract': [1, {'$mod': [x, 1]}]}]}
    elif isinstance(t, tuple(datetime_terms)):
        op = datetime_terms[type(t)]
        return {'$%s' % op: compute_sub(t._child)}
    elif isinstance(t, UTCFromTimestamp):
        return {'$add': [datetime.datetime.utcfromtimestamp(0),
                         {'$multiply': [1000, compute_sub(t._child)]}]}
    raise NotImplementedError('Operation %s not supported' % type(t).__name__)


@dispatch((Projection, Field), MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$project': dict((col, 1) for col in t.fields)})


@dispatch(SimpleSelection, MongoQuery)
def compute_up(expr, data, **kwargs):
    predicate = optimize(expr.predicate, data)

    if not isinstance(predicate, Broadcast):
        raise TypeError("Selection predicate must be a broadcastable "
                        "operation.\nReceived an expression of type %r.\n"
                        "Available broadcastable operations are (%s)" %
                        (type(predicate).__name__,
                         ', '.join(map(compose(repr, attrgetter('__name__')),
                                   Broadcastable))))

    s = predicate._scalars[0]
    d = dict((s[c], symbol(c, s[c].dshape.measure)) for c in s.fields)
    expr = predicate._scalar_expr._subs(d)
    return data.append({'$match': match(expr)})


@dispatch(By, MongoQuery)
def compute_up(t, q, **kwargs):
    if not isinstance(t.grouper, (Field, Projection, Symbol)):
        raise ValueError("Complex By operations not supported on MongoDB.\n"
                "The grouping element must be a simple Field or Projection\n"
                "Got %s" % t.grouper)
    apply = optimize(t.apply, q)
    names = apply.fields
    return MongoQuery(q.coll, q.query +
    ({
        '$group': toolz.merge(
                    {'_id': dict((col, '$'+col) for col in t.grouper.fields)},
                    group_apply(apply)
                    )
     },
     {
         '$project': toolz.merge(dict((col, '$_id.'+col) for col in t.grouper.fields),
                                 dict((name, '$' + name) for name in names))
     }))


@dispatch(nunique, MongoQuery)
def compute_up(t, q, **kwargs):
    return MongoQuery(q.coll, q.query +
    ({'$group': {'_id': dict((col, '$'+col) for col in t.fields),
                 t._name: {'$sum': 1}}},
     {'$project': {'_id': 0, t._name: 1}}))


@dispatch(Distinct, MongoQuery)
def compute_up(t, q, **kwargs):
    return MongoQuery(q.coll, q.query +
    ({'$group': {'_id': dict((col, '$'+col) for col in t.fields)}},
     {'$project': toolz.merge(dict((col, '$_id.'+col) for col in t.fields),
                              {'_id': 0})}))


@dispatch(Reduction)
def group_apply(expr):
    """
    Dictionary corresponding to apply part of split-apply-combine operation

    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> group_apply(accounts.amount.sum())
    {'amount_sum': {'$sum': '$amount'}}
    """
    key = expr._name
    col = '$' + expr._child._name
    if isinstance(expr, count):
        return {key: {'$sum': 1}}
    if isinstance(expr, sum):
        return {key: {'$sum': col}}
    if isinstance(expr, max):
        return {key: {'$max': col}}
    if isinstance(expr, min):
        return {key: {'$min': col}}
    if isinstance(expr, mean):
        return {key: {'$avg': col}}
    raise NotImplementedError("Reduction %s not yet supported in MongoDB"
                              % type(expr).__name__)


reductions = {mean: '$avg', count: '$sum', max: '$max', min: '$min', sum: '$sum'}


def scalar_expr(expr):
    if isinstance(expr, Broadcast):
        s = expr._scalars[0]
        d = dict((s[c], symbol(c, s[c].dshape.measure)) for c in s.fields)
        return expr._scalar_expr._subs(d)
    elif isinstance(expr, Field):
        return symbol(expr._name, expr.dshape.measure)
    else:
        # TODO: This is a hack
        # broadcast_collect should reach into summary, By, selection
        # And perform these kinds of optimizations itself
        expr2 = broadcast_collect(expr)
        if not expr2.isidentical(expr):
            return scalar_expr(expr2)


@dispatch(Summary)
def group_apply(expr):
    # TODO: implement columns variable more generally when Broadcast works
    reducs = expr.values
    names = expr.fields
    values = [(name, c, getattr(c._child, 'column', None) or name)
               for name, c in zip(names, reducs)]

    query = dict((k, {reductions[type(v)]: 1 if isinstance(v, count)
                                        else compute_sub(scalar_expr(v._child))})
                 for k, v, z in values)
    return query


@dispatch((count, nelements), MongoQuery)
def compute_up(t, q, **kwargs):
    name = t._name
    return q.append({'$group': {'_id': {}, name: {'$sum': 1}}})


@dispatch((sum, min, max, mean), MongoQuery)
def compute_up(t, q, **kwargs):
    name = t._name
    reduction = {sum: '$sum', min: '$min', max: '$max', mean: '$avg'}[type(t)]
    column = '$' + t._child._name
    arg = {'$group': {'_id': {}, name: {reduction: column}}}
    return q.append(arg)


@dispatch(Sort, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$sort': {t.key: 1 if t.ascending else -1}})


datetime_terms = {Day: 'dayOfMonth',
                  Month: 'month',
                  Year: 'year',
                  Minute: 'minute',
                  Second: 'second'}


@dispatch(Field, Database)
def compute_up(expr, data, **kwargs):
    return getattr(data, expr._name)


@dispatch(Expr, Collection)
def post_compute(e, c, scope=None):
    """
    Calling compute on a raw collection?  Compute on an empty MongoQuery.
    """
    return post_compute(e, MongoQuery(c, ()), scope=scope)


def get_result(result):
    try:
        return result['result']
    except TypeError:
        return list(result)


@dispatch(Expr, MongoQuery)
def post_compute(e, q, scope=None):
    """
    Execute a query using MongoDB's aggregation pipeline

    The compute_up functions operate on Mongo Collection / list-of-dict
    queries.  Once they're done we need to actually execute the query on
    MongoDB.  We do this using the aggregation pipeline framework.

    http://docs.mongodb.org/manual/core/aggregation-pipeline/
    """
    scope = {'$project': toolz.merge({'_id': 0},  # remove mongo identifier
                                 dict((col, 1) for col in e.fields))}
    q = q.append(scope)

    if not e.dshape.shape:  # not a collection
        result = get_result(q.coll.aggregate(list(q.query)))[0]
        if isscalar(e.dshape.measure):
            return result[e._name]
        else:
            return get(e.fields, result)

    dicts = get_result(q.coll.aggregate(list(q.query)))

    if isscalar(e.dshape.measure):
        return list(pluck(e.fields[0], dicts, default=None))  # dicts -> values
    else:
        return list(pluck(e.fields, dicts, default=None))  # dicts -> tuples


@dispatch(Broadcast, MongoQuery)
def post_compute(e, q, scope=None):
    """Compute the result of a Broadcast expression.
    """
    columns = dict((col, 1) for qry in q.query
                   for col in qry.get('$project', []))
    scope = {'$project': toolz.merge({'_id': 0},  # remove mongo identifier
                                 dict((col, 1) for col in columns))}
    q = q.append(scope)
    dicts = get_result(q.coll.aggregate(list(q.query)))

    assert len(columns) == 1
    return list(pluck(first(columns.keys()), dicts))


def name(e):
    """

    >>> name(Symbol('x', 'int32'))
    'x'
    >>> name(1)
    1
    """
    if isinstance(e, Symbol):
        return e._name
    elif isinstance(e, Expr):
        raise NotImplementedError("Complex queries not yet supported")
    else:
        return e


opnames = {
    Lt: '$lt',
    Le: '$lte',
    Gt: '$gt',
    Ge: '$gte',
    Ne: '$ne',
    And: '$and',
    Or: '$or',
    Like: '$regex',
}


@dispatch((Lt, Le, Gt, Ge, Ne))
def match(expr):
    return {name(expr.lhs): {opnames[type(expr)]: expr.rhs}}


@dispatch((And, Or))
def match(expr):
    return {opnames[type(expr)]: [match(expr.lhs), match(expr.rhs)]}


@dispatch(Eq)
def match(expr):
    return {name(expr.lhs): name(expr.rhs)}


@dispatch(Like)
def match(expr):
    return {
        name(expr._child): {
            opnames[type(expr)]: fnmatch.translate(expr.pattern)
        }
    }


@dispatch(object)
def match(expr):
    """ Match query for MongoDB

    Examples
    --------
    >>> x = symbol('x', 'int32')
    >>> name = symbol('name', 'string')
    >>> match(name == 'Alice')
    {'name': 'Alice'}
    >>> match(x > 10)
    {'x': {'$gt': 10}}
    >>> match(10 > x)
    {'x': {'$lt': 10}}
    >>> match((x > 10) & (name == 'Alice'))  # doctest: +SKIP
    {'x': {'$gt': 10}, 'name': 'Alice'}
    >>> match((x > 10) | (name == 'Alice'))
    {'$or': [{'x': {'$gt': 10}}, {'name': 'Alice'}]}
    """
    raise NotImplementedError(
        'Matching not supported on expressions of type %r' % type(expr).__name__
    )
