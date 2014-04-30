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

base = (int, float, str, bool)

@dispatch(Projection, sqlalchemy.Table)
def _compute(t, s):
    s = _compute(t.table, s)
    return sa.select([s.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.Table)
def _compute(t, s):
    s = _compute(t.table, s)
    return s.c.get(t.columns[0])


@dispatch(base, object)
def _compute(a, b):
    return a


@dispatch(ColumnWise, sqlalchemy.Table)
def _compute(t, s):
    return t.op(_compute(t.lhs, s), _compute(t.rhs, s))


@dispatch(Selection, sqlalchemy.Table)
def _compute(t, s):
    return sa.select([_compute(t.table, s)]).where(_compute(t.predicate, s))


@dispatch(TableExpr, sqlalchemy.Table)
def _compute(t, s):
    return s

@dispatch(TableExpr, sqlalchemy.Table)
def compute(t, s):
    result = _compute(t, s)
    if not isinstance(result, sqlalchemy.sql.selectable.Select):
        result = sa.select([result])
    return result
