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

from multipledispatch import dispatch
import itertools
from collections import Iterator
import math
from operator import itemgetter
from functools import partial

from ..expr.table import *
from ..compatibility import builtins
from ..utils import groupby, get, reduceby

Sequence = (tuple, list, Iterator)

@dispatch(Projection, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    indices = [t.parent.columns.index(col) for col in t.columns]
    get = operator.itemgetter(*indices)
    return (get(x) for x in parent)


@dispatch(Column, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    index = t.parent.columns.index(t.columns[0])
    return (x[index] for x in parent)


@dispatch(BinOp, Sequence)
def compute(t, seq):
    lhs_istable = isinstance(t.lhs, TableExpr)
    rhs_istable = isinstance(t.rhs, TableExpr)

    if lhs_istable and rhs_istable:

        seq1, seq2 = itertools.tee(seq, 2)
        lhs = compute(t.lhs, seq1)
        rhs = compute(t.rhs, seq2)

        return (t.op(left, right) for left, right in zip(lhs, rhs))

    elif lhs_istable:

        lhs = compute(t.lhs, seq)
        right = compute(t.rhs, None)

        return (t.op(left, right) for left in lhs)

    elif rhs_istable:

        rhs = compute(t.rhs, seq)
        left = compute(t.lhs, None)

        return (t.op(left, right) for right in rhs)


@dispatch(Selection, Sequence)
def compute(t, seq):
    seq1, seq2 = itertools.tee(seq)
    parent = compute(t.parent, seq1)
    predicate = compute(t.predicate, seq2)
    return (x for x, tf in zip(parent, predicate)
              if tf)


@dispatch(TableSymbol, Sequence)
def compute(t, seq):
    return seq


@dispatch(UnaryOp, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    op = getattr(math, t.symbol)
    return (op(x) for x in parent)

@dispatch(Reduction, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
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

@dispatch(count, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return builtins.sum(1 for i in parent)

@dispatch(distinct, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return set(parent)

@dispatch(nunique, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return len(set(parent))

@dispatch(mean, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return _mean(parent)

@dispatch(var, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return _var(parent)

@dispatch(std, Sequence)
def compute(t, seq):
    return math.sqrt(compute(var(t.parent), seq))

lesser = lambda x, y: x if x < y else y
greater = lambda x, y: x if x > y else y
countit = lambda acc, _: acc + 1

binops = {sum: (operator.add, 0),
          min: (lesser, 1e250),
          max: (greater, -1e250),
          count: (countit, 0),
          any: (operator.or_, False),
          all: (operator.and_, True)}

@dispatch(By, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)

    if isinstance(t.grouper, Projection) and t.grouper.parent == t.parent:
        indices = [t.grouper.parent.columns.index(col)
                        for col in t.grouper.columns]
        grouper = operator.itemgetter(*indices)
    else:
        raise NotImplementedError("Grouper attribute of By must be Projection "
                                  "of parent table, got %s" % str(t.grouper))

    # Match setting like
    # By(t, t[column], t[column].reduction())
    # TODO: Support more general streaming grouped reductions
    if (isinstance(t.apply, Reduction) and
        isinstance(t.apply.parent, Column) and
        t.apply.parent.parent.isidentical(t.grouper.parent) and
        t.apply.parent.parent.isidentical(t.parent) and
        type(t.apply) in binops):

        binop, initial = binops[type(t.apply)]

        col = t.apply.parent.columns[0]
        getter = operator.itemgetter(t.apply.parent.parent.columns.index(col))
        def binop2(acc, x):
            x = getter(x)
            return binop(acc, x)

        d = reduceby(grouper, binop2, parent, initial)
    else:
        groups = groupby(grouper, parent)
        d = dict((k, compute(t.apply, v)) for k, v in groups.items())
    return d.items()


@dispatch(Join, Sequence, Sequence)
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
    get_right = lambda x: type(x)(get(right_columns, x))

    lhs_dict = groupby(partial(get, left_index), lhs)

    for row in rhs:
        try:
            key = row[right_index]
            matches = lhs_dict[key]
            for match in matches:
                yield match + get_right(row)
        except KeyError:
            pass


@dispatch(Sort, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    if isinstance(t.column, (tuple, list)):
        index = [t.parent.columns.index(col) for col in t.column]
        key = operator.itemgetter(*index)
    else:
        index = t.parent.columns.index(t.column)
        key = operator.itemgetter(index)

    return sorted(parent,
                  key=key,
                  reverse=not t.ascending)


@dispatch(Head, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return itertools.islice(parent, 0, t.n)


@dispatch((Label, ReLabel), Sequence)
def compute(t, seq):
    return compute(t.parent, seq)


@dispatch(Map, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    if len(t.parent.columns) == 1:
        return map(t.func, parent)
    else:
        return itertools.starmap(t.func, parent)


@dispatch(Apply, Sequence)
def compute(t, seq):
    parent = compute(t.parent, seq)
    return t.func(parent)
