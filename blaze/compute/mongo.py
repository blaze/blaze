""" MongoDB Backend - Uses aggregation pipeline

If you don't have a mongo server running

    $ conda install mongodb -y
    $ mongod &

>>> from blaze import *

>>> data = [(1, 'Alice', 100),
...         (2, 'Bob', -200),
...         (3, 'Charlie', 300),
...         (4, 'Denis', 400),
...         (5, 'Edith', -500)]

Migrate data into MongoDB

>>> import pymongo
>>> db = pymongo.MongoClient().db
>>> _ = into(db.mydata, data, columns=['id', 'name', 'amount'])

Objective: find the name of accounts with negative amount

Using MongoDB query language

>>> db.mydata.aggregate([{'$match': {'amount': {'$lt': 0}}}, # doctest: +SKIP
...                      {'$project': {'name': 1, '_id': 0}}])['result']
[{'name': 'Bob'}, {'name': 'Edith'}]

Using Blaze

>>> t = Table(db.mydata)
>>> t[t.amount < 0].name
    name
0    Bob
1  Edith

>>> db.mydata.drop()

Uses the aggregation pipeline
http://docs.mongodb.org/manual/core/aggregation-pipeline/
"""

from __future__ import absolute_import, division, print_function

from datashape import discover, isdimension, dshape
from collections import Iterator
import pymongo
from toolz import take, concat, partition_all, pluck
import toolz
from pymongo.collection import Collection

from ..expr.table import *
from ..expr.core import Expr

from ..dispatch import dispatch


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


@dispatch((var, Label, std, Sort, count, nunique, Selection, mean, Reduction, Head, ReLabel,
    Apply, Distinct, RowWise,  By), Collection)
def compute_one(e, coll, **kwargs):
    return compute_one(e, MongoQuery(coll, []))


@dispatch(TableSymbol, Collection)
def compute_one(t, coll, **kwargs):
    return MongoQuery(coll, [])


@dispatch(Head, MongoQuery)
def compute_one(t, q, **kwargs):
    return q.append({'$limit': t.n})


@dispatch(Projection, MongoQuery)
def compute_one(t, q, **kwargs):
    return q.append({'$project': dict((col, 1) for col in t.columns)})


@dispatch(Selection, MongoQuery)
def compute_one(t, q, **kwargs):
    return q.append({'$match': match(t.predicate.expr)})


@dispatch(By, MongoQuery)
def compute_one(t, q, **kwargs):
    if not (isinstance(t.grouper, Projection) and t.grouper.child == t.child):
        raise ValueError("Complex By operations not supported on MongoDB.\n"
                "Must be of the form `by(t, t[columns], t[column].reduction()`")
    name = t.apply.dshape[0].names[0]
    return MongoQuery(q.coll, q.query +
    ({
        '$group': toolz.merge(
                    {'_id': dict((col, '$'+col) for col in t.grouper.columns)},
                    group_apply(t.apply)
                    )
     },
     {
         '$project': toolz.merge(dict((col, '$_id.'+col) for col in t.grouper.columns),
                                 {name: '$'+name})
     }))


def group_apply(expr):
    """
    Dictionary corresponding to apply part of split-apply-combine operation

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> group_apply(accounts.amount.sum())
    {'amount_sum': {'$sum': '$amount'}}
    """
    assert isinstance(expr.dshape[0], Record)
    key = expr.dshape[0].names[0]
    col = '$'+expr.child.columns[0]
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
    raise NotImplementedError("Only certain reductions supported in MongoDB")


@dispatch(count, MongoQuery)
def compute_one(t, q, **kwargs):
    name = t.dshape[0].names[0]
    return q.append({'$group': {'_id': {}, name: {'$sum': 1}}})


@dispatch((sum, min, max, mean), MongoQuery)
def compute_one(t, q, **kwargs):
    name = t.dshape[0].names[0]
    reduction = {sum: '$sum', min: '$min', max: '$max', mean: '$avg'}[type(t)]
    column = '$' + t.child.columns[0]
    return q.append({'$group': {'_id': {}, name: {reduction: column}}})


@dispatch(Sort, MongoQuery)
def compute_one(t, q, **kwargs):
    return q.append({'$sort': {t.key: 1 if t.ascending else -1}})


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
    """
    q = q.append({'$project': toolz.merge({'_id': 0},
                                      dict((col, 1) for col in e.columns))})
    dicts = q.coll.aggregate(list(q.query))['result']

    if isinstance(e, TableExpr) and e.iscolumn:
        return list(pluck(e.columns[0], dicts))
    if isinstance(e, TableExpr):
        return list(pluck(e.columns, dicts))


def name(e):
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
