"""

>>> from blaze.expr.table import TableExpr
>>> from blaze.compute.python import compute

>>> accounts = TableExpr('{name: string, amount: int}')
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
    s = compute(t.table, s)
    return sa.select([s.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.Table)
def compute(t, s):
    s = compute(t.table, s)
    return s.c.get(t.columns[0])


@dispatch(BinOp, sqlalchemy.Table)
def compute(t, s):
    return t.op(compute(t.lhs, s), compute(t.rhs, s))


@dispatch(Selection, sqlalchemy.Table)
def compute(t, s):
    return sa.select([compute(t.table, s)]).where(compute(t.predicate, s))


@dispatch(TableExpr, sqlalchemy.Table)
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
