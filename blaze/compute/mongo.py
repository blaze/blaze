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


class query(object):
    def __init__(self, coll, query):
        self.coll = coll
        self.query = tuple(query)

    def append(self, clause):
        return query(self.coll, self.query + (clause,))


    def info(self):
        return self.coll, self.query

    def __eq__(self, other):
        return type(self) == type(other) and self.info() == other.info()

    def __hash__(self):
        return hash((type(self), self.info()))

@dispatch((var, Label, std, Sort, count, nunique, Selection, mean, Reduction, Head, ReLabel,
    Apply, Distinct, RowWise,  By), Collection)
def compute_one(e, coll, **kwargs):
    return compute_one(e, query(coll, []))


@dispatch(TableSymbol, Collection)
def compute_one(t, coll, **kwargs):
    return query(coll, [])


@dispatch(Head, query)
def compute_one(t, q, **kwargs):
    return q.append({'$limit': t.n})


@dispatch(Projection, query)
def compute_one(t, q, **kwargs):
    return q.append({'$project': dict((col, 1) for col in t.columns)})


@dispatch(Selection, query)
def compute_one(t, q, **kwargs):
    return q.append({'$match': match(t.predicate.expr)})


@dispatch(Expr, Collection, dict)
def post_compute(e, c, d):
    return post_compute(e, query(c, ()), d)


@dispatch(Expr, query, dict)
def post_compute(e, q, d):
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
    >>> match((x > 10) & (name == 'Alice'))
    {'x': {'$gt': 10}, 'name': 'Alice'}
    >>> match((x > 10) | (name == 'Alice'))
    {'$or': [{'x': {'$gt': 10}}, {'name': 'Alice'}]}
    """
    if isinstance(expr, Eq):
        return {name(expr.lhs): name(expr.rhs)}
    if isinstance(expr, Lt):
        if not isinstance(expr.lhs, Expr):
            return match(expr.rhs > expr.lhs)
        return {name(expr.lhs): {'$lt': name(expr.rhs)}}
    if isinstance(expr, Gt):
        if not isinstance(expr.lhs, Expr):
            return match(expr.rhs < expr.lhs)
        return {name(expr.lhs): {'$gt': name(expr.rhs)}}
    if isinstance(expr, And):
        return toolz.merge(match(expr.lhs), match(expr.rhs))
    if isinstance(expr, Or):
        return {'$or': [match(expr.lhs), match(expr.rhs)]}
