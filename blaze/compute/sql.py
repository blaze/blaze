"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.sql import compute

>>> accounts = TableSymbol('{name: string, amount: int}')
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

from blaze.expr.table import *
from blaze.utils import unique
from multipledispatch import dispatch
import sqlalchemy as sa
import sqlalchemy

@dispatch(Projection, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return sa.select([parent.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    return parent.c.get(t.columns[0])


@dispatch(BinOp, sqlalchemy.sql.Selectable)
def compute(t, s):
    lhs = compute(t.lhs, s)
    rhs = compute(t.rhs, s)
    return t.op(lhs, rhs)


@dispatch(Selection, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    predicate = compute(t.predicate, s)
    return sa.select([parent]).where(predicate)


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


@dispatch(Join, sqlalchemy.sql.Selectable, sqlalchemy.sql.Selectable)
def compute(t, lhs, rhs):
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)

    left_column = getattr(lhs.c, t.on_left)
    right_column = getattr(rhs.c, t.on_right)

    condition = left_column == right_column

    join = lhs.join(rhs, condition)

    columns = unique(join.columns,
                     key=lambda c: c.name)
    return select(list(columns)).select_from(join)


@dispatch(UnaryOp, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    op = getattr(sa.func, t.symbol)
    return op(parent)


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
    return op(parent)


@dispatch(By, sqlalchemy.sql.Selectable)
def compute(t, s):
    parent = compute(t.parent, s)
    if isinstance(t.grouper, Projection):
        grouper = [compute(t.grouper.parent[col], s) for col in t.grouper.columns]
    else:
        raise NotImplementedError("Grouper must be a projection, got %s"
                                  % t.grouper)
    return select(compute(t.apply, s)).group_by(*grouper)


@dispatch(Sort, sqlalchemy.sql.Selectable)
def compute(t, s):
    if isinstance(t.column, (tuple, list)):
        raise NotImplementedError("Multi-column sort not yet implemented")
    parent = compute(t.parent, s)
    col = getattr(parent.c, t.column)
    if not t.ascending:
        col = sqlalchemy.desc(col)
    return select(parent).order_by(col)
