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
from .core import compute_one, compute, base

__all__ = ['compute', 'compute_one', 'computefull', 'select']

@dispatch(Projection, Selectable)
def compute_one(t, s, scope={}, **kwargs):
    # Walk up the tree to get the original columns
    subexpression = t
    while hasattr(subexpression, 'child'):
        subexpression = subexpression.child
    csubexpression = compute(subexpression, scope)
    # Hack because csubexpression may be SQL object
    if not isinstance(csubexpression, Selectable):
        csubexpression = csubexpression.table
    columns = [csubexpression.c.get(col) for col in t.columns]

    return select(s).with_only_columns(columns)


@dispatch(Column, Selectable)
def compute_one(t, s, **kwargs):
    return s.c.get(t.columns[0])


@dispatch(ColumnWise, Selectable)
def compute_one(t, s, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
    d = dict((t.child[c].scalar_symbol, getattr(s.c, c)) for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(BinOp, ClauseElement, (ClauseElement, base))
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (ClauseElement, base), ClauseElement)
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, ClauseElement)
def compute_one(t, s, **kwargs):
    op = getattr(sa.func, t.symbol)
    return op(s)


@dispatch(USub, (sa.Column, Selectable))
def compute_one(t, s, **kwargs):
    return -s


@dispatch(Selection, Selectable)
def compute_one(t, s, **kwargs):
    predicate = compute(t.predicate, {t.child: s})
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
def compute_one(t, lhs, rhs, **kwargs):
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
def compute_one(t, s, **kwargs):
    try:
        op = getattr(sqlalchemy.sql.functions, t.symbol)
    except:
        symbol = names.get(type(t), t.symbol)
        op = getattr(sqlalchemy.sql.func, symbol)
    result = op(s)

    if isinstance(t.child.schema[0], Record):
        name = list(t.child.schema[0].fields.keys())[0]
        result = result.label(name)

    return result


@dispatch(nunique, ClauseElement)
def compute_one(t, s, **kwargs):
    return sqlalchemy.sql.functions.count(sqlalchemy.distinct(s))


@dispatch(Distinct, (sa.Column, Selectable))
def compute_one(t, s, **kwargs):
    return sqlalchemy.distinct(s)


@dispatch(By, Selectable)
def compute_one(t, s, **kwargs):
    if isinstance(t.grouper, Projection):
        grouper = [compute(t.grouper.child[col], {t.child: s})
                    for col in t.grouper.columns]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    reduction = compute(t.apply, {t.child: s})
    return select(grouper + [reduction]).group_by(*grouper)


@dispatch(Sort, Selectable)
def compute_one(t, s, **kwargs):
    if isinstance(t.key, (tuple, list)):
        raise NotImplementedError("Multi-column sort not yet implemented")
    col = getattr(s.c, t.key)
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return select(s).order_by(col)


@dispatch(Head, ClauseElement)
def compute_one(t, s, **kwargs):
    if hasattr(s, 'limit'):
        return s.limit(t.n)
    else:
        return select(s).limit(t.n)


@dispatch(Label, ClauseElement)
def compute_one(t, s, **kwargs):
    return s.label(t.label)


@dispatch(ReLabel, Selectable)
def compute_one(t, s, **kwargs):
    columns = [getattr(s.c, col).label(new_col)
               if col != new_col else
               getattr(s.c, col)
               for col, new_col in zip(t.child.columns, t.columns)]

    return select(columns)


@dispatch(Merge, Selectable)
def compute_one(t, s, **kwargs):
    subexpression = common_subexpression(*t.children)
    children = [compute(child, {subexpression: s}) for child in t.children]
    return select(children)
