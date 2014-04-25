"""

>>> from blaze.objects.table import Table
>>> from blaze.compute.python import compute

>>> accounts = Table('{name: string, amount: int}')
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

from blaze.objects.table import *
from multipledispatch import dispatch
import sqlalchemy as sa
import sqlalchemy

base = (int, float, str, bool)

@dispatch(Projection, sqlalchemy.Table)
def compute(t, s):
    s = compute(t.table, s)
    return sa.select([s.c.get(col) for col in t.columns])


@dispatch(Column, sqlalchemy.Table)
def compute(t, s):
    s = compute(t.table, s)
    return s.c.get(t.columns[0])


@dispatch(base, object)
def compute(a, b):
    return a


@dispatch(Relational, sqlalchemy.Table)
def compute(t, s):
    return t.op(compute(t.lhs, s), compute(t.rhs, s))


@dispatch(Selection, sqlalchemy.Table)
def compute(t, s):
    return sa.select([compute(t.table, s)]).where(compute(t.predicate, s))


@dispatch(Table, sqlalchemy.Table)
def compute(t, s):
    return s
