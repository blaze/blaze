"""

>>> from blaze import *

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> from sqlalchemy import Table, Column, MetaData, Integer, String
>>> t = Table('accounts', MetaData(),
...           Column('name', String, primary_key = True),
...           Column('amount', Integer))
>>> print(compute(deadbeats, t))  # doctest: +SKIP
SELECT accounts.name
FROM accounts
WHERE accounts.amount < :amount_1
"""
from __future__ import absolute_import, division, print_function
import sqlalchemy as sa
import sqlalchemy
from sqlalchemy import sql
from sqlalchemy.sql import Selectable
from sqlalchemy.sql.elements import ClauseElement
from operator import and_

from ..dispatch import dispatch
from ..expr.table import *
from ..expr.scalar import BinOp, UnaryOp
from ..compatibility import reduce
from ..utils import unique
from . import core
from .core import compute_one, compute, base

__all__ = ['compute', 'compute_one', 'computefull', 'select']

@dispatch(Projection, Selectable)
def compute_one(t, s):
    # Walk up the tree to get the original columns
    ancestor = t
    while hasattr(ancestor, 'parent'):
        ancestor = ancestor.parent
    ancestor = compute(ancestor, s)
    columns = [ancestor.c.get(col) for col in t.columns]

    return select(s).with_only_columns(columns)


@dispatch(Column, Selectable)
def compute_one(t, s):
    return s.c.get(t.columns[0])


@dispatch(ColumnWise, Selectable)
def compute_one(t, s):
    columns = [t.parent[c] for c in t.parent.columns]
    d = dict((t.parent[c].scalar_symbol, getattr(s.c, c)) for c in t.parent.columns)
    return compute(t.expr, d)


@dispatch(BinOp, ClauseElement, (ClauseElement, base))
def compute_one(t, lhs, rhs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (ClauseElement, base), ClauseElement)
def compute_one(t, lhs, rhs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, ClauseElement)
def compute_one(t, s):
    op = getattr(sa.func, t.symbol)
    return op(s)


@dispatch(Neg, (sa.Column, Selectable))
def compute_one(t, s):
    return -s


@dispatch(Selection, Selectable)
def compute_one(t, s):
    predicate = compute(t.predicate, {t.parent: s})
    try:
        return s.where(predicate)
    except AttributeError:
        return select([s]).where(predicate)


def select(s):
    """ Permissive SQL select

    Idempotent sqlalchemy.select

    Wraps input in list if neccessary
    """
    if not isinstance(s, sqlalchemy.sql.Select):
        if not isinstance(s, (tuple, list)):
            s = [s]
        s = sa.select(s)
    return s


def computefull(t, s):
    return select(compute(t, s))


def listpack(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


@dispatch(Join, Selectable, Selectable)
def compute_one(t, lhs, rhs):
    condition = reduce(and_, [getattr(lhs.c, l) == getattr(rhs.c, r)
        for l, r in zip(listpack(t.on_left), listpack(t.on_right))])

    join = lhs.join(rhs, condition)

    columns = unique(join.columns,
                     key=lambda c: c.name)
    return select(list(columns)).select_from(join)



names = {mean: 'avg',
         var: 'variance',
         std: 'stdev'}


@dispatch(Reduction, sql.elements.ClauseElement)
def compute_one(t, s):
    try:
        op = getattr(sqlalchemy.sql.functions, t.symbol)
    except:
        symbol = names.get(type(t), t.symbol)
        op = getattr(sqlalchemy.sql.func, symbol)
    result = op(s)

    if isinstance(t.parent.schema[0], Record):
        name = list(t.parent.schema[0].fields.keys())[0]
        result = result.label(name)

    return result


@dispatch(nunique, ClauseElement)
def compute_one(t, s):
    return sqlalchemy.sql.functions.count(sqlalchemy.distinct(s))


@dispatch(Distinct, (sa.Column, Selectable))
def compute_one(t, s):
    return sqlalchemy.distinct(s)


@dispatch(By, Selectable)
def compute_one(t, s):
    if isinstance(t.grouper, Projection):
        grouper = [compute(t.grouper.parent[col], {t.parent: s})
                    for col in t.grouper.columns]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    reduction = compute(t.apply, {t.parent: s})
    return select(grouper + [reduction]).group_by(*grouper)


@dispatch(Sort, Selectable)
def compute_one(t, s):
    if isinstance(t.column, (tuple, list)):
        raise NotImplementedError("Multi-column sort not yet implemented")
    col = getattr(s.c, t.column)
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return select(s).order_by(col)


@dispatch(Head, ClauseElement)
def compute_one(t, s):
    return select(s).limit(t.n)


@dispatch(Label, ClauseElement)
def compute_one(t, s):
    return s.label(t.label)


@dispatch(ReLabel, Selectable)
def compute_one(t, s):
    columns = [getattr(s.c, col).label(new_col)
               if col != new_col else
               getattr(s.c, col)
               for col, new_col in zip(t.parent.columns, t.columns)]

    return select(columns)


@dispatch(Merge, Selectable)
def compute_one(t, s):
    ancestor = common_ancestor(*t.children)
    children = [compute(child, {ancestor: s}) for child in t.children]
    return select(children)
