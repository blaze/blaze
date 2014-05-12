"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.python import compute

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
from multipledispatch import dispatch
import sqlalchemy as sa
import sqlalchemy

@dispatch(Projection, sqlalchemy.Table)
def compute(t, s):
    parent = compute(t.parent, s)
    return sa.select([parent.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.Table)
def compute(t, s):
    parent = compute(t.parent, s)
    return parent.c.get(t.columns[0])


@dispatch(BinOp, sqlalchemy.Table)
def compute(t, s):
    lhs = compute(t.lhs, s)
    rhs = compute(t.rhs, s)
    return t.op(lhs, rhs)


@dispatch(Selection, sqlalchemy.Table)
def compute(t, s):
    parent = compute(t.parent, s)
    predicate = compute(t.predicate, s)
    return sa.select([parent]).where(predicate)


@dispatch(TableSymbol, sqlalchemy.Table)
def compute(t, s):
    return s


def computefull(t, s):
    result = compute(t, s)
    if not isinstance(result, sqlalchemy.sql.selectable.Select):
        result = sa.select([result])
    return result


@dispatch(Join, sqlalchemy.Table, sqlalchemy.Table)
def compute(t, lhs, rhs):
    """

    TODO: SQL bunches all of the columns from both tables together.  We should
    probably downselect
    """
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)

    left_column = getattr(lhs.c, t.on_left)
    right_column = getattr(rhs.c, t.on_right)

    return lhs.join(rhs, left_column == right_column)


@dispatch(UnaryOp, sqlalchemy.Table)
def compute(t, s):
    parent = compute(t.parent, s)
    op = getattr(sa.func, t.symbol)
    return op(parent)


names = {mean: 'avg',
         var: 'variance',
         std: 'stdev'}


@dispatch(Reduction, sqlalchemy.Table)
def compute(t, s):
    parent = compute(t.parent, s)
    try:
        op = getattr(sqlalchemy.sql.functions, t.symbol)
    except:
        symbol = names.get(type(t), t.symbol)
        op = getattr(sqlalchemy.sql.func, symbol)
    return op(parent)
