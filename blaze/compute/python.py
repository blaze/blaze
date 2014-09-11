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
import numbers
import fnmatch
import re
from collections import Iterator
from functools import partial
from toolz import map, filter, compose, juxt, identity
from cytoolz import groupby, reduceby, unique, take, concat, first
import cytoolz
import toolz
import sys
import math

from ..dispatch import dispatch
from ..expr.table import TableSymbol
from ..expr.table import (Projection, Column, ColumnWise, Map, Label, ReLabel,
                          Merge, RowWise, Join, Selection, Reduction, Distinct,
                          By, Sort, Head, Apply, Union, Summary, Like)
from ..expr.table import count, nunique, mean, var, std
from ..expr import table, eval_str
from ..expr.scalar.numbers import BinOp, UnaryOp, RealMath
from ..compatibility import builtins, apply, unicode
from . import core
from .core import compute, compute_one

from ..data import DataDescriptor
from ..data.utils import listpack

# Dump exp, log, sin, ... into namespace
import math
from math import *


__all__ = ['compute', 'compute_one', 'Sequence', 'rowfunc', 'rrowfunc']

Sequence = (tuple, list, Iterator, type(dict().items()))


def recursive_rowfunc(t, stop):
    """ Compose rowfunc functions up a tree

    >>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
    >>> expr = accounts['amount'].map(lambda x: x + 1)
    >>> f = recursive_rowfunc(expr, accounts)

    >>> row = ('Alice', 100)
    >>> f(row)
    101

    """
    funcs = []
    while not t.isidentical(stop):
        funcs.append(rowfunc(t))
        t = t.child
    return compose(*funcs)


rrowfunc = recursive_rowfunc


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
    from cytoolz.curried import get
    indices = [t.child.columns.index(col) for col in t.columns]
    return get(indices)


@dispatch(Column)
def rowfunc(t):
    if t.child.iscolumn and t.column == t.child.columns[0]:
        return identity
    index = t.child.columns.index(t.column)
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
    if t.child.iscolumn:
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
    funcs = [rrowfunc(child, t.child) for child in t.children]
    return compose(concat_maybe_tuples, juxt(*funcs))


@dispatch(RowWise, Sequence)
def compute_one(t, seq, **kwargs):
    return map(rowfunc(t), seq)


@dispatch(Selection, Sequence)
def compute_one(t, seq, **kwargs):
    predicate = rrowfunc(t.predicate, t.child)
    return filter(predicate, seq)


@dispatch(Reduction, Sequence)
def compute_one(t, seq, **kwargs):
    op = getattr(builtins, t.symbol)
    return op(seq)


@dispatch(BinOp, numbers.Real, numbers.Real)
def compute_one(bop, a, b, **kwargs):
    return bop.op(a, b)


@dispatch(UnaryOp, numbers.Real)
def compute_one(uop, x, **kwargs):
    return uop.op(x)


@dispatch(RealMath, numbers.Real)
def compute_one(f, n, **kwargs):
    return getattr(math, type(f).__name__)(n)


def _mean(seq):
    total = 0
    count = 0
    for item in seq:
        total += item
        count += 1
    return float(total) / count


def _var(seq, unbiased):
    total = 0
    total_squared = 0
    count = 0
    for item in seq:
        total += item
        total_squared += item * item
        count += 1

    return (total_squared - (total * total) / count) / (count - unbiased)


def _std(seq, unbiased):
    return math.sqrt(_var(seq, unbiased))


@dispatch(count, Sequence)
def compute_one(t, seq, **kwargs):
    return cytoolz.count(seq)


@dispatch(Distinct, Sequence)
def compute_one(t, seq, **kwargs):
    try:
        row = first(seq)
    except StopIteration:
        return ()
    seq = concat([[row], seq]) # re-add row to seq

    if isinstance(row, list):
        seq = map(tuple, seq)

    return unique(seq)


@dispatch(nunique, Sequence)
def compute_one(t, seq, **kwargs):
    return len(set(seq))


@dispatch(mean, Sequence)
def compute_one(t, seq, **kwargs):
    return _mean(seq)


@dispatch(var, Sequence)
def compute_one(t, seq, **kwargs):
    return _var(seq, t.unbiased)


@dispatch(std, Sequence)
def compute_one(t, seq, **kwargs):
    return _std(seq, t.unbiased)


lesser = lambda x, y: x if x < y else y
greater = lambda x, y: x if x > y else y
countit = lambda acc, _: acc + 1


from operator import add, or_, and_

# Dict mapping
# Reduction : (binop, combiner, init)

# Reduction :: [a] -> b
# binop     :: b, a -> b
# combiner  :: b, b -> b
# init      :: b
binops = {table.sum: (add, add, 0),
          table.min: (lesser, lesser, 1e250),
          table.max: (greater, lesser, -1e250),
          table.count: (countit, add, 0),
          table.any: (or_, or_, False),
          table.all: (and_, and_, True)}


def reduce_by_funcs(t):
    """ Create grouping func and binary operator for a by-reduction/summary

    Turns a by operation like

        by(t.name, t.amount.sum())

    into a grouper like

    >>> def grouper(row):
    ...     return row[name_index]

    and a binary operator like

    >>> def binop(acc, row):
    ...     return binops[sum](acc, row[amount_index])

    It also handles this in the more complex ``summary`` case in which case
    several binary operators are juxtaposed together.

    See Also:
        compute_one(By, Sequence)
    """
    grouper = rrowfunc(t.grouper, t.child)
    if (isinstance(t.apply, Reduction) and
        type(t.apply) in binops):

        binop, combiner, initial = binops[type(t.apply)]
        applier = rrowfunc(t.apply.child, t.child)

        def binop2(acc, x):
            return binop(acc, applier(x))

        return grouper, binop2, combiner, initial

    elif (isinstance(t.apply, Summary) and
        builtins.all(type(val) in binops for val in t.apply.values)):

        binops2, combiners, inits = zip(*[binops[type(v)] for v in t.apply.values])
        appliers = [rrowfunc(v.child, t.child) for v in t.apply.values]

        def binop2(accs, x):
            return tuple(binop(acc, applier(x)) for binop, acc, applier in
                        zip(binops2, accs, appliers))

        def combiner(a, b):
            return tuple(c(x, y) for c, x, y in zip(combiners, a, b))

        return grouper, binop2, combiner, tuple(inits)


@dispatch(By, Sequence)
def compute_one(t, seq, **kwargs):
    if ((isinstance(t.apply, Reduction) and type(t.apply) in binops) or
        (isinstance(t.apply, Summary) and builtins.all(type(val) in binops
                                                for val in t.apply.values))):
        grouper, binop, combiner, initial = reduce_by_funcs(t)
        d = reduceby(grouper, binop, seq, initial)
    else:
        grouper = rrowfunc(t.grouper, t.child)
        groups = groupby(grouper, seq)
        d = dict((k, compute(t.apply, {t.child: v})) for k, v in groups.items())

    if t.grouper.iscolumn:
        keyfunc = lambda x: (x,)
    else:
        keyfunc = identity
    if isinstance(t.apply, Reduction):
        valfunc = lambda x: (x,)
    else:
        valfunc = identity
    return tuple(keyfunc(k) + valfunc(v) for k, v in d.items())


def pair_assemble(t):
    """ Combine a pair of records into a single record

    This is mindful to shared columns as well as missing records
    """
    from cytoolz import get  # not curried version
    on_left = [t.lhs.columns.index(col) for col in listpack(t.on_left)]
    on_right = [t.rhs.columns.index(col) for col in listpack(t.on_right)]

    left_self_columns = [t.lhs.columns.index(c) for c in t.lhs.columns
                                            if c not in listpack(t.on_left)]
    right_self_columns = [t.rhs.columns.index(c) for c in t.rhs.columns
                                            if c not in listpack(t.on_right)]
    def assemble(pair):
        a, b = pair
        if a is not None:
            joined = get(on_left, a)
        else:
            joined = get(on_right, b)

        if a is not None:
            left_entries = get(left_self_columns, a)
        else:
            left_entries = (None,) * (len(t.lhs.columns) - len(on_left))

        if b is not None:
            right_entries = get(right_self_columns, b)
        else:
            right_entries = (None,) * (len(t.rhs.columns) - len(on_right))

        return joined + left_entries + right_entries

    return assemble


@dispatch(Join, (DataDescriptor, Sequence), (DataDescriptor, Sequence))
def compute_one(t, lhs, rhs, **kwargs):
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

    on_left = [t.lhs.columns.index(col) for col in listpack(t.on_left)]
    on_right = [t.rhs.columns.index(col) for col in listpack(t.on_right)]

    left_default = (None if t.how in ('right', 'outer')
                         else toolz.itertoolz.no_default)
    right_default = (None if t.how in ('left', 'outer')
                         else toolz.itertoolz.no_default)

    pairs = toolz.join(on_left, lhs,
                       on_right, rhs,
                       left_default=left_default,
                       right_default=right_default)

    assemble = pair_assemble(t)

    return map(assemble, pairs)


@dispatch(Sort, Sequence)
def compute_one(t, seq, **kwargs):
    if isinstance(t.key, (str, unicode, tuple, list)):
        key = rowfunc(t.child[t.key])
    else:
        key = rowfunc(t.key)
    return sorted(seq,
                  key=key,
                  reverse=not t.ascending)


@dispatch(Head, Sequence)
def compute_one(t, seq, **kwargs):
    if t.n < 100:
        return tuple(take(t.n, seq))
    else:
        return take(t.n, seq)


@dispatch((Label, ReLabel), Sequence)
def compute_one(t, seq, **kwargs):
    return seq


@dispatch(Apply, Sequence)
def compute_one(t, seq, **kwargs):
    return t.func(seq)


@dispatch(Union, Sequence, tuple)
def compute_one(t, example, children, **kwargs):
    return concat(children)


@dispatch(Summary, Sequence)
def compute_one(expr, data, **kwargs):
    if isinstance(data, Iterator):
        datas = itertools.tee(data, len(expr.values))
        return tuple(compute(val, {expr.child: data})
                        for val, data in zip(expr.values, datas))
    else:
        return tuple(compute(val, {expr.child: data})
                        for val in expr.values)


@dispatch(BinOp, object, object)
def compute_one(expr, x, y, **kwargs):
    return expr.op(x, y)


@dispatch(UnaryOp, object)
def compute_one(expr, x, **kwargs):
    return eval(eval_str(expr), toolz.merge(locals(), math.__dict__))


def like_regex_predicate(expr):
    regexes = dict((name, re.compile('^' + fnmatch.translate(pattern) + '$'))
                    for name, pattern in expr.patterns.items())
    regex_tup = [regexes.get(name, None) for name in expr.columns]
    def predicate(tup):
        for item, regex in zip(tup, regex_tup):
            if regex and not regex.match(item):
                return False
        return True

    return predicate
@dispatch(Like, Sequence)
def compute_one(expr, seq, **kwargs):
    predicate = like_regex_predicate(expr)
    return filter(predicate, seq)
