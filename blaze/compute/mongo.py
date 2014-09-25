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

>> t = Table(db.mydata)
>> t[t.amount < 0].name
    name
0    Bob
1  Edith

>> db.mydata.drop()

Uses the aggregation pipeline
http://docs.mongodb.org/manual/core/aggregation-pipeline/
"""

from __future__ import absolute_import, division, print_function

import operator as op
import numbers

try:
    from pymongo.collection import Collection
except ImportError:
    Collection = type(None)

import fnmatch
from datashape import Record
from toolz import pluck, first
import toolz

from ..expr import (var, Label, std, Sort, count, nunique, Selection, mean,
                    Reduction, Head, ReLabel, Apply, Distinct, RowWise, By,
                    TableSymbol, Projection, sum, min, max, TableExpr,
                    Gt, Lt, Ge, Le, Eq, Ne, ScalarSymbol, And, Or, Summary,
                    Like, Arithmetic, ColumnWise, DateTime, Microsecond, Date,
                    Time)
from ..expr.core import Expr
from ..compatibility import _strtypes

from ..dispatch import dispatch


__all__ = ['MongoQuery']


class MongoQuery(object):
    """
    A Pair of a pymongo collection and a aggregate query

    We need to carry around both a pymongo collection and a list of
    dictionaries to feed into the aggregation pipeline.  This class
    carries around such a pair.

    Parameters
    ----------
    coll: pymongo.collection.Collection
        A single pymongo collection, holds a table
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


@dispatch((var, Label, std, Sort, count, nunique, Selection, mean, Reduction,
           Head, ReLabel, Apply, Distinct, RowWise, By, Like, DateTime),
          Collection)
def compute_up(e, coll, **kwargs):
    return compute_up(e, MongoQuery(coll, []))


@dispatch(TableSymbol, Collection)
def compute_up(t, coll, **kwargs):
    return MongoQuery(coll, [])


@dispatch(Head, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$limit': t.n})


@dispatch(ColumnWise, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$project': {str(t.expr): compute_sub(t.expr)}})


binops = {'+': 'add',
          '*': 'multiply',
          '/': 'divide',
          '-': 'subtract',
          '%': 'mod'}


def compute_sub(t):
    """Build an expression tree in a MongoDB compatible way.

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
    >>> from blaze import ScalarSymbol
    >>> s = ScalarSymbol('s', 'float64')
    >>> expr = s * 2 + s - 1
    >>> expr
    ((s * 2) + s) - 1
    >>> compute_sub(expr)
    {'$subtract': [{'$add': [{'$multiply': ['$s', 2]}, '$s']}, 1]}
    """
    if isinstance(t, (_strtypes, ScalarSymbol)):
        return '$%s' % t
    elif isinstance(t, numbers.Number):
        return t
    try:
        op = binops[t.symbol]
    except KeyError:
        raise NotImplementedError('Arithmetic operation %r not implemented in '
                                  'MongoDB' % t.symbol)
    return {compute_sub(op): [compute_sub(t.lhs), compute_sub(t.rhs)]}


@dispatch(Projection, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$project': dict((col, 1) for col in t.columns)})


@dispatch(Selection, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$match': match(t.predicate.expr)})


@dispatch(Like, MongoQuery)
def compute_up(t, q, **kwargs):
    pats = dict((name, {'$regex': fnmatch.translate(pattern)})
                for name, pattern in t.patterns.items())
    return q.append({'$match': pats})


@dispatch(By, MongoQuery)
def compute_up(t, q, **kwargs):
    if not isinstance(t.grouper, Projection):
        raise ValueError("Complex By operations not supported on MongoDB.\n"
                "Must be of the form `by(t[columns], t[column].reduction()`")
    names = t.apply.dshape[0].names
    return MongoQuery(q.coll, q.query +
    ({
        '$group': toolz.merge(
                    {'_id': dict((col, '$'+col) for col in t.grouper.columns)},
                    group_apply(t.apply)
                    )
     },
     {
         '$project': toolz.merge(dict((col, '$_id.'+col) for col in t.grouper.columns),
                                 dict((name, '$' + name) for name in names))
     }))


@dispatch(Distinct, MongoQuery)
def compute_up(t, q, **kwargs):
    return MongoQuery(q.coll, q.query +
    ({'$group': {'_id': dict((col, '$'+col) for col in t.columns)}},
     {'$project': toolz.merge(dict((col, '$_id.'+col) for col in t.columns),
                              {'_id': 0})}))


@dispatch(Reduction)
def group_apply(expr):
    """
    Dictionary corresponding to apply part of split-apply-combine operation

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> group_apply(accounts.amount.sum())
    {'amount_sum': {'$sum': '$amount'}}
    """
    assert isinstance(expr.dshape[0], Record)
    key = expr.dshape[0].names[0]
    col = '$' + expr.child.columns[0]
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


reductions = {mean: 'avg', count: 'sum', max: 'max', min: 'min'}


@dispatch(Summary)
def group_apply(expr):
    # TODO: implement columns variable more generally when ColumnWise works
    reducs = expr.values
    names = expr.names
    values = [(name, c, getattr(c.child, 'column', None) or name)
               for name, c in zip(names, reducs)]
    key_getter = lambda v: '$%s' % reductions.get(type(v), type(v).__name__)
    query = dict((k, {key_getter(v): int(isinstance(v, count)) or
                      compute_sub(v.child.expr)})
                 for k, v, z in values)
    return query


@dispatch(count, MongoQuery)
def compute_up(t, q, **kwargs):
    name = t.dshape[0].names[0]
    return q.append({'$group': {'_id': {}, name: {'$sum': 1}}})


@dispatch((sum, min, max, mean), MongoQuery)
def compute_up(t, q, **kwargs):
    name = t.dshape[0].names[0]
    reduction = {sum: '$sum', min: '$min', max: '$max', mean: '$avg'}[type(t)]
    column = '$' + t.child.columns[0]
    arg = {'$group': {'_id': {}, name: {reduction: column}}}
    return q.append(arg)


@dispatch(Sort, MongoQuery)
def compute_up(t, q, **kwargs):
    return q.append({'$sort': {t.key: 1 if t.ascending else -1}})


@dispatch(DateTime, MongoQuery)
def compute_up(expr, q, **kwargs):
    attr = expr.attr
    d = {'$project': {expr.column:
                      {'$%s' % {'day': 'dayOfMonth'}.get(attr, attr):
                       '$%s' % expr.child.column}}}
    return q.append(d)


@dispatch((Date, Time, Microsecond), MongoQuery)
def compute_up(expr, q, **kwargs):
    raise NotImplementedError('MongoDB does not support the %r field' %
                              type(expr).__name__.lower())


@dispatch(Expr, Collection, dict)
def post_compute(e, c, d):
    """
    Calling compute on a raw collection?  Compute on an empty MongoQuery.
    """
    return post_compute(e, MongoQuery(c, ()), d)


@dispatch(Expr, MongoQuery, dict)
def post_compute(e, q, d):
    """
    Get single result, like a sum or count, from mongodb query
    """
    field = e.dshape[0].names[0]
    result = q.coll.aggregate(list(q.query))['result']
    return result[0][field]


@dispatch(TableExpr, MongoQuery, dict)
def post_compute(e, q, d):
    """
    Execute a query using MongoDB's aggregation pipeline

    The compute_up functions operate on Mongo Collection / list-of-dict
    queries.  Once they're done we need to actually execute the query on
    MongoDB.  We do this using the aggregation pipeline framework.

    http://docs.mongodb.org/manual/core/aggregation-pipeline/
    """
    d = {'$project': toolz.merge({'_id': 0},  # remove mongo identifier
                                 dict((col, 1) for col in e.columns))}
    q = q.append(d)
    dicts = q.coll.aggregate(list(q.query))['result']

    if e.iscolumn:
        return list(pluck(e.columns[0], dicts, default=None))  # dicts -> values
    else:
        return list(pluck(e.columns, dicts, default=None))  # dicts -> tuples


@dispatch(ColumnWise, MongoQuery, dict)
def post_compute(e, q, d):
    """Compute the result of a columnwise expression.
    """
    columns = dict((col, 1) for qry in q.query
                   for col in qry.get('$project', []))
    d = {'$project': toolz.merge({'_id': 0},  # remove mongo identifier
                                 dict((col, 1) for col in columns))}
    q = q.append(d)
    dicts = q.coll.aggregate(list(q.query))['result']

    assert len(columns) == 1
    return list(pluck(first(columns.keys()), dicts))


def name(e):
    """

    >>> name(ScalarSymbol('x', 'int32'))
    'x'
    >>> name(1)
    1
    """
    if isinstance(e, ScalarSymbol):
        return e._name
    elif isinstance(e, Expr):
        raise NotImplementedError("Complex queries not yet supported")
    else:
        return e


# Reflective binary operator, e.g. (x < y) -> (y > x)
binop_swap = {Lt: Gt, Gt: Lt, Ge: Le, Le: Ge, Eq: Eq, Ne: Ne}


def match(expr):
    """ Match query for MongoDB

    Examples
    --------
    >>> x = ScalarSymbol('x', 'int32')
    >>> name = ScalarSymbol('name', 'string')
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
    if not isinstance(expr.lhs, Expr):
        return match(binop_swap(type(expr))(expr.lhs, expr.rhs))
    if isinstance(expr, Eq):
        return {name(expr.lhs): name(expr.rhs)}
    if isinstance(expr, Lt):
        return {name(expr.lhs): {'$lt': expr.rhs}}
    if isinstance(expr, Le):
        return {name(expr.lhs): {'$lte': expr.rhs}}
    if isinstance(expr, Gt):
        return {name(expr.lhs): {'$gt': expr.rhs}}
    if isinstance(expr, Ge):
        return {name(expr.lhs): {'$gte': expr.rhs}}
    if isinstance(expr, And):
        return toolz.merge(match(expr.lhs), match(expr.rhs))
    if isinstance(expr, Or):
        return {'$or': [match(expr.lhs), match(expr.rhs)]}
    if isinstance(expr, Ne):
        return {name(expr.lhs): {'$ne': expr.rhs}}
    raise NotImplementedError("Matching not supported on expressions of type %s"
            % type(expr).__name__)
