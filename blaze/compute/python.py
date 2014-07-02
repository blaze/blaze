""" Python compute layer

>>> from blaze import *
>>> from blaze.compute.core import compute

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> list(compute(deadbeats, data))
['Bob', 'Charlie']
"""
from __future__ import absolute_import, division, print_function

import itertools
from collections import Iterator
import math
from operator import itemgetter
from functools import partial
from toolz import map, isiterable, compose, juxt, identity
from toolz.compatibility import zip
import sys

from ..dispatch import dispatch
from ..expr.table import *
from ..expr.scalar.core import *
from ..expr import scalar
from ..compatibility import builtins, apply
from cytoolz import groupby, get, reduceby, unique, take
import cytoolz
from . import core
from .core import compute, compute_one

from ..data import DataDescriptor

# Dump exp, log, sin, ... into namespace
from math import *

__all__ = ['compute', 'compute_one', 'Sequence']

Sequence = (tuple, list, Iterator)



def recursive_rowfunc(t):
    """ Compose rowfunc functions up a tree

    Stops when we hit a non-RowWise operation

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> f = recursive_rowfunc(accounts['amount'].map(lambda x: x + 1))

    >>> row = ('Alice', 100)
    >>> f(row)
    101

    """
    funcs = []
    while isinstance(t, RowWise):
        funcs.append(rowfunc(t))
        t = t.parent
    if not funcs:
        raise TypeError("Expected RowWise operation, got %s" % str(t))
    elif len(funcs) == 1:
        return funcs[0]
    else:
        return compose(*funcs)

@dispatch(TableSymbol)
def rowfunc(t):
    return identity

@dispatch(Projection)
def rowfunc(t):
    """ Rowfunc provides a function that can be mapped onto a sequence.

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> f = rowfunc(accounts['amount'])

    >>> row = ('Alice', 100)
    >>> f(row)
    100

    See Also:
        compute<Rowwise, Sequence>
    """
    from toolz.curried import get
    indices = [t.parent.columns.index(col) for col in t.columns]
    return get(indices)


@dispatch(Column)
def rowfunc(t):
    if t.parent.iscolumn and t.column == t.parent.columns[0]:
        return identity
    index = t.parent.columns.index(t.column)
    return lambda x: x[index]


@dispatch(ColumnWise)
def rowfunc(t):
    if sys.version_info[0] == 3:
        # Python3 doesn't allow argument unpacking
        # E.g. ``lambda (x, y, z): x + z`` is illegal
        # Solution: Make ``lambda x, y, z: x + y``, then wrap with ``apply``
        func = eval(core.columnwise_funcstr(t, variadic=True, full=True))
        return partial(apply, func)
    elif sys.version_info[0] == 2:
        return eval(core.columnwise_funcstr(t, variadic=False, full=True))


@dispatch(Map)
def rowfunc(t):
    if len(t.parent.columns) == 1:
        return t.func
    else:
        return partial(apply, t.func)


@dispatch((Label, ReLabel))
def rowfunc(t):
    return identity


def concat_maybe_tuples(vals):
    """

    >>> concat_maybe_tuples([1, (2, 3)])
    (1, 2, 3)
    """
    result = []
    for v in vals:
        if isinstance(v, (tuple, list)):
            result.extend(v)
        else:
            result.append(v)
    return tuple(result)


@dispatch(Merge)
def rowfunc(t):
    funcs = list(map(recursive_rowfunc, t.children))
    return compose(concat_maybe_tuples, juxt(*funcs))


@dispatch(RowWise, Sequence)
def compute_one(t, seq):
    return map(rowfunc(t), seq)


@dispatch(Selection, Sequence)
def compute_one(t, seq):
    return filter(rowfunc(t.predicate), seq)


@dispatch(Reduction, Sequence)
def compute_one(t, seq):
    op = getattr(builtins, t.symbol)
    return op(seq)


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


def _std(seq):
    return sqrt(_var(seq))


@dispatch(count, Sequence)
def compute_one(t, seq):
    return cytoolz.count(seq)


@dispatch(Distinct, Sequence)
def compute_one(t, seq):
    return unique(seq)


@dispatch(nunique, Sequence)
def compute_one(t, seq):
    return len(set(seq))


@dispatch(mean, Sequence)
def compute_one(t, seq):
    return _mean(seq)


@dispatch(var, Sequence)
def compute_one(t, seq):
    return _var(seq)


@dispatch(std, Sequence)
def compute_one(t, seq):
    return _std(seq)


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
def compute_one(t, seq):
    if (isinstance(t.apply, Reduction) and
        type(t.apply) in binops):

        binop, initial = binops[type(t.apply)]
        applier = rowfunc(t.apply.parent.subs({t.apply.parent.parent:
                                               t.parent}))
        grouper = rowfunc(t.grouper.subs({t.grouper.parent: t.parent}))

        def binop2(acc, x):
            return binop(acc, applier(x))

        d = reduceby(grouper, binop2, seq, initial)
    else:
        grouper = rowfunc(t.grouper)
        groups = groupby(grouper, seq)
        d = dict((k, compute(t.apply, v)) for k, v in groups.items())

    if t.grouper.iscolumn:
        return d.items()
    else:
        return tuple(k + (v,) for k, v in d.items())

@dispatch(Join, Sequence)
def compute_one(t, seq):
    a, b = itertools.tee(seq)
    return compute(t, a, b)


def listpack(x):
    """

    >>> listpack(1)
    [1]
    >>> listpack((1, 2))
    [1, 2]
    >>> listpack([1, 2])
    [1, 2]
    """
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    else:
        return [x]



@dispatch(Join, (DataDescriptor, Sequence), (DataDescriptor, Sequence))
def compute_one(t, lhs, rhs):
    """ Join Operation for Python Streaming Backend

    Note that a pure streaming Join is challenging/impossible because any row
    in one seq might connect to any row in the other, requiring simultaneous
    complete access.

    As a result this approach compromises and fully realizes the LEFT sequence
    while allowing the RIGHT sequence to stream.  As a result

    Always put your bigger table on the RIGHT side of the Join.
    """
    if lhs == rhs:
        lhs, rhs = itertools.tee(lhs, 2)

    on_left = rowfunc(t.lhs[t.on_left])
    on_right = rowfunc(t.rhs[t.on_right])

    right_columns = list(range(len(t.rhs.columns)))
    for col in listpack(t.on_right):
        right_columns.remove(t.rhs.columns.index(col))

    get_right = lambda x: type(x)(get(right_columns, x))

    lhs_dict = groupby(on_left, lhs)

    for row in rhs:
        try:
            key = on_right(row)
            matches = lhs_dict[key]
            for match in matches:
                yield match + get_right(row)
        except KeyError:
            pass


@dispatch(Sort, Sequence)
def compute_one(t, seq):
    if isinstance(t.key, (str, tuple, list)):
        key = rowfunc(t.parent[t.key])
    else:
        key = rowfunc(t.key)
    return sorted(seq,
                  key=key,
                  reverse=not t.ascending)


@dispatch(Head, Sequence)
def compute_one(t, seq):
    return tuple(take(t.n, seq))


@dispatch((Label, ReLabel), Sequence)
def compute_one(t, seq):
    return seq


@dispatch(Apply, Sequence)
def compute_one(t, seq):
    return t.func(seq)
