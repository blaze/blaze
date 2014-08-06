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


@dispatch(By, query)
def compute_one(t, q, **kwargs):
    if not (isinstance(t.grouper, Projection) and t.grouper.child == t.child):
        raise ValueError("Complex By operations not supported on MongoDB.\n"
                "Must be of the form `by(t, t[columns], t[column].reduction()`")
    name = t.apply.dshape[0].names[0]
    return query(q.coll, q.query +
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


@dispatch(count, query)
def compute_one(t, q, **kwargs):
    name = t.dshape[0].names[0]
    return q.append({'$group': {'_id': {}, name: {'$sum': 1}}})


@dispatch((sum, min, max, mean), query)
def compute_one(t, q, **kwargs):
    name = t.dshape[0].names[0]
    reduction = {sum: '$sum', min: '$min', max: '$max', mean: '$avg'}[type(t)]
    column = '$' + t.child.columns[0]
    return q.append({'$group': {'_id': {}, name: {reduction: column}}})


@dispatch(Sort, query)
def compute_one(t, q, **kwargs):
    return q.append({'$sort': {t.key: 1 if t.ascending else -1}})


@dispatch(Expr, Collection, dict)
def post_compute(e, c, d):
    return post_compute(e, query(c, ()), d)


@dispatch(Expr, query, dict)
def post_compute(e, q, d):
    name = e.dshape[0].names[0]
    return q.coll.aggregate(list(q.query))['result'][0][name]


@dispatch(TableExpr, query, dict)
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
