"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.sql import compute

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]

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
from operator import and_

from ..dispatch import dispatch
from ..expr.table import *
from ..expr.scalar import BinOp, UnaryOp
from ..compatibility import reduce
from ..utils import unique
from . import core

__all__ = ['compute', 'computefull', 'select']

@dispatch(Projection, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    # Walk up the tree to get the original columns
    ancestor = t
    while hasattr(ancestor, 'parent'):
        ancestor = ancestor.parent
    ancestor = compute(ancestor, s)
    columns = [ancestor.c.get(col) for col in t.columns]

    return select(parent).with_only_columns(columns)


@dispatch(Column, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return parent.c.get(t.columns[0])


@dispatch(ColumnWise, sqlalchemy.sql.Selectable)
def compute(t, s):
    expr = t.expr
    columns = [t.parent[c] for c in t.parent.columns]
    expr = expr.subs(dict((col.scalar_symbol, col) for col in columns))
    return compute(expr, s)


@dispatch(BinOp, sqlalchemy.sql.Selectable)
def compute(t, s):
    lhs = compute(t.lhs, s)
    rhs = compute(t.rhs, s)
    return t.op(lhs, rhs)


@dispatch(UnaryOp, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    op = getattr(sa.func, t.symbol)
    return op(parent)


@dispatch(Neg, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return -parent


@dispatch(Selection, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    predicate = compute(t.predicate, s)
    try:
        return parent.where(predicate)
    except AttributeError:
        return select([parent]).where(predicate)


@dispatch(TableSymbol, sqlalchemy.sql.Selectable)
def compute(t, s):
    return s


def select(s):
    """ Permissive SQL select

    Idempotent sqlalchemy.select

    Wraps input in list if neccessary
    """
    if not isinstance(s, sqlalchemy.sql.selectable.Select):
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


@dispatch(Join, sqlalchemy.sql.Selectable, sqlalchemy.sql.Selectable)
def compute(t, lhs, rhs):
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)

    condition = reduce(and_, [getattr(lhs.c, l) == getattr(rhs.c, r)
        for l, r in zip(listpack(t.on_left), listpack(t.on_right))])

    join = lhs.join(rhs, condition)

    columns = unique(join.columns,
                     key=lambda c: c.name)
    return select(list(columns)).select_from(join)



names = {mean: 'avg',
         var: 'variance',
         std: 'stdev'}


@dispatch(Reduction, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    try:
        op = getattr(sqlalchemy.sql.functions, t.symbol)
    except:
        symbol = names.get(type(t), t.symbol)
        op = getattr(sqlalchemy.sql.func, symbol)
    result = op(parent)

    if isinstance(t.parent.schema[0], Record):
        name = list(t.parent.schema[0].fields.keys())[0]
        result = result.label(name)

    return result


@dispatch(nunique, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)

    return sqlalchemy.sql.functions.count(sqlalchemy.distinct(parent))


@dispatch(Distinct, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return sqlalchemy.distinct(parent)


@dispatch(By, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    if isinstance(t.grouper, Projection):
        grouper = [compute(t.grouper.parent[col], parent)
                    for col in t.grouper.columns]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    reduction = compute(t.apply, parent)
    return select(grouper + [reduction]).group_by(*grouper)


@dispatch(Sort, sqlalchemy.sql.Selectable)
def compute(t, s):
    if isinstance(t.column, (tuple, list)):
        raise NotImplementedError("Multi-column sort not yet implemented")
    parent = compute(t.parent, s)
    col = getattr(parent.c, t.column)
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return select(parent).order_by(col)


@dispatch(Head, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return select(parent).limit(t.n)


@dispatch(Label, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return parent.label(t.label)


@dispatch(ReLabel, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)

    columns = [getattr(s.c, col).label(new_col)
               if col != new_col else
               getattr(s.c, col)
               for col, new_col in zip(t.parent.columns, t.columns)]

    return select(columns)


@dispatch(Merge, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    children = [compute(child, parent) for child in t.children]
    return select(children)
