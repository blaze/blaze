"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.python import compute

>>> accounts = TableSymbol('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]

>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> list(compute(deadbeats, data))
['Bob', 'Charlie']
"""
from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from blaze.compatibility import builtins
from blaze.utils import groupby, get
from multipledispatch import dispatch
import itertools
from collections import Iterator
import math

seq = (tuple, list, Iterator)

@dispatch(Projection, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    indices = [t.parent.columns.index(col) for col in t.columns]
    get = operator.itemgetter(*indices)
    return (get(x) for x in parent)


@dispatch(Column, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    index = t.parent.columns.index(t.columns[0])
    return (x[index] for x in parent)


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
    l1, l2 = itertools.tee(l)
    parent = compute(t.parent, l1)
    predicate = compute(t.predicate, l2)
    return (x for x, tf in zip(parent, predicate)
              if tf)


@dispatch(TableSymbol, seq)
def compute(t, l):
    return l


@dispatch(UnaryOp, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    op = getattr(math, t.symbol)
    return (op(x) for x in parent)

@dispatch(Reduction, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    op = getattr(builtins, t.symbol)
    return op(parent)

def _mean(seq):
    total = 0
    count = 0
    for item in seq:
        total += item
        count += 1
    return float(total) / count

def _var(seq):
    total = 0
    total_squared = 0
    count = 0
    for item in seq:
        total += item
        total_squared += item ** 2
        count += 1
    return 1.0*total_squared/count - (1.0*total/count) ** 2

@dispatch(count, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    return builtins.sum(1 for i in parent)

@dispatch(mean, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    return _mean(parent)

@dispatch(var, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    return _var(parent)

@dispatch(std, seq)
def compute(t, l):
    return math.sqrt(compute(var(t.parent), l))


@dispatch(By, seq)
def compute(t, l):
    parent = compute(t.parent, l)

    if isinstance(t.grouper, Projection) and t.grouper.parent == t.parent:
        indices = [t.grouper.parent.columns.index(col)
                        for col in t.grouper.columns]
        grouper = operator.itemgetter(*indices)
    else:
        raise NotImplementedError("Grouper attribute of By must be Projection "
                                  "of parent table, got %s" % str(t.grouper))

    groups = groupby(grouper, parent)
    d = dict((k, compute(t.apply, v)) for k, v in groups.items())
    return d.items()


@dispatch(Join, seq, seq)
def compute(t, lhs, rhs):
    """ Join Operation for Python Streaming Backend

    Note that a pure streaming Join is challenging/impossible because any row
    in one seq might connect to any row in the other, requiring simultaneous
    complete access.

    As a result this approach compromises and fully realizes the LEFT sequence
    while allowing the RIGHT sequence to stream.  As a result

    Always put your bigger table on the RIGHT side of the Join.
    """
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)

    left_index = t.lhs.columns.index(t.on_left)
    right_index = t.rhs.columns.index(t.on_right)

    right_columns = list(range(len(t.rhs.columns)))
    right_columns.remove(right_index)
    get_right = lambda x: get(right_columns, x)

    lhs_dict = dict((row[left_index], row) for row in lhs)

    return (lhs_dict[row[right_index]] + get_right(row) for row in rhs)


@dispatch(Sort, seq)
def compute(t, l):
    parent = compute(t.parent, l)
    if isinstance(t.column, (tuple, list)):
        index = [t.parent.columns.index(col) for col in t.column]
        key = operator.itemgetter(*index)
    else:
        index = t.parent.columns.index(t.column)
        key = operator.itemgetter(index)

    return sorted(parent,
                  key=key,
                  reverse=not t.ascending)
