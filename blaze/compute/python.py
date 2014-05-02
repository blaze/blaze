"""

>>> from blaze.expr.table import TableExpr
>>> from blaze.compute.python import compute

>>> accounts = TableExpr('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]

>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> list(compute(deadbeats, data))
['Bob', 'Charlie']
"""
from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from multipledispatch import dispatch
import itertools
from collections import Iterator

seq = (tuple, list, Iterator)

@dispatch(Projection, seq)
def compute(t, l):
    indices = [t.table.columns.index(col) for col in t.columns]
    get = operator.itemgetter(*indices)
    return (get(x) for x in l)


@dispatch(Column, seq)
def compute(t, l):
    index = t.table.columns.index(t.columns[0])
    return (x[index] for x in l)


@dispatch(BinOp, seq)
def compute(t, l):
    lhs_istable = isinstance(t.lhs, TableExpr)
    rhs_istable = isinstance(t.rhs, TableExpr)

    if lhs_istable and rhs_istable:

        l1, l2 = itertools.tee(l, 2)
        lhs = compute(t.lhs, l1)
        rhs = compute(t.rhs, l2)

        return (t.op(left, right) for left, right in zip(lhs, rhs))

    elif lhs_istable:

        lhs = compute(t.lhs, l)
        right = compute(t.rhs, None)

        return (t.op(left, right) for left in lhs)

    elif rhs_istable:

        rhs = compute(t.rhs, l)
        left = compute(t.lhs, None)

        return (t.op(left, right) for right in rhs)


@dispatch(Selection, seq)
def compute(t, l):
    l, l2 = itertools.tee(l)
    return (x for x, tf in zip(compute(t.table, l), compute(t.predicate, l2))
              if tf)


@dispatch(TableExpr, seq)
def compute(t, l):
    return l
