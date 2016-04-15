""" Python compute layer

>>> from blaze import symbol, compute

>>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> list(compute(deadbeats, data))
['Bob', 'Charlie']
"""
from __future__ import absolute_import, division, print_function

from collections import Mapping
import itertools
import numbers
import fnmatch
import operator
import datetime
import math
import random

from collections import Iterator
from functools import partial

from datashape import Option, to_numpy_dtype
import numpy as np
import pandas as pd
import toolz
from toolz import map, filter, compose, juxt, identity, tail

try:
    from cytoolz import groupby, reduceby, unique, take, concat, nth, pluck
except ImportError:
    from toolz import groupby, reduceby, unique, take, concat, nth, pluck

from datashape.predicates import isscalar, iscollection

from ..dispatch import dispatch
from ..expr import (Projection, Field, Broadcast, Map, Label, ReLabel,
                    Merge, Join, Selection, Reduction, Distinct,
                    By, Sort, Head, Sample, Apply, Summary, Like, IsIn,
                    DateTime, Date, Time, Millisecond, ElemWise,
                    Symbol, Slice, Expr, Arithmetic, ndim, DateTimeTruncate,
                    UTCFromTimestamp, notnull, UnaryMath, greatest, least,
                    Coerce)
from ..expr import reductions
from ..expr import count, nunique, mean, var, std
from ..expr import BinOp, UnaryOp, USub, Not, nelements
from ..compatibility import builtins, apply, unicode, _inttypes
from .core import compute, compute_up, optimize, base

from ..utils import listpack
from ..expr.broadcast import broadcast_collect
from .pyfunc import lambdify, funcstr
from . import pydatetime

# Dump exp, log, sin, ... into namespace
from math import (
    atan2,
    atanh,
    ceil,
    copysign,
    cos,
    cosh,
    degrees,
    e,
    erf,
    erfc,
    exp,
    expm1,
    fabs,
    factorial,
    floor,
    fmod,
    frexp,
    fsum,
    gamma,
    hypot,
    isinf,
    isnan,
    ldexp,
    lgamma,
    log,
    log10,
    log1p,
    modf,
    pi,
    pow,
    radians,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
)



__all__ = ['compute', 'compute_up', 'Sequence', 'rowfunc', 'rrowfunc']

Sequence = (tuple, list, Iterator, type(dict().items()))


@dispatch(Expr, Sequence)
def pre_compute(expr, seq, scope=None, **kwargs):
    try:
        if isinstance(seq, Iterator):
            first = next(seq)
            seq = concat([[first], seq])
        else:
            first = next(iter(seq))
    except StopIteration:
        return []
    if isinstance(first, Mapping):
        leaf = expr._leaves()[0]
        return pluck(leaf.fields, seq)
    else:
        return seq


@dispatch(Expr, Sequence)
def optimize(expr, seq):
    return broadcast_collect(expr)


def child(x):
    if hasattr(x, '_child'):
        return x._child
    if hasattr(x, '_inputs') and len(x._inputs) == 1:
        return x._inputs[0]
    raise NotImplementedError("Found expression with multiple children.\n"
                              "Perhaps Broadcast Optimize was not called")


def recursive_rowfunc(t, stop):
    """ Compose rowfunc functions up a tree

    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> expr = accounts['amount'].map(lambda x: x + 1)
    >>> f = recursive_rowfunc(expr, accounts)

    >>> row = ('Alice', 100)
    >>> f(row)
    101

    """
    funcs = []
    while not t.isidentical(stop):
        funcs.append(rowfunc(t))
        t = child(t)
    return compose(*funcs)


rrowfunc = recursive_rowfunc


@dispatch(Symbol)
def rowfunc(t):
    return identity


@dispatch(Projection)
def rowfunc(t):
    """ Rowfunc provides a function that can be mapped onto a sequence.

    >>> from blaze import symbol
    >>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
    >>> f = rowfunc(accounts['amount'])

    >>> row = ('Alice', 100)
    >>> f(row)
    100

    See Also:
        compute<Rowwise, Sequence>
    """
    from toolz.itertoolz import getter
    indices = [t._child.fields.index(col) for col in t.fields]
    return getter(indices)


@dispatch(Field)
def rowfunc(t):
    index = t._child.fields.index(t._name)
    return lambda x, index=index: x[index]


@dispatch(IsIn)
def rowfunc(t):
    return t._keys.__contains__


@dispatch(Broadcast)
def rowfunc(t):
    return lambdify(t._scalars, t._scalar_expr)


@dispatch(Arithmetic)
def rowfunc(expr):
    return eval(funcstr(expr))


@dispatch(Map)
def rowfunc(t):
    if isscalar(t._child.dshape.measure):
        return t.func
    else:
        return partial(apply, t.func)


@dispatch((Label, ReLabel))
def rowfunc(t):
    return identity


@dispatch(DateTime)
def rowfunc(t):
    return lambda row: getattr(row, t.attr)


@dispatch(UTCFromTimestamp)
def rowfunc(t):
    return datetime.datetime.utcfromtimestamp


@dispatch((Date, Time))
def rowfunc(t):
    return lambda row: getattr(row, t.attr)()


@dispatch(Millisecond)
def rowfunc(_):
    return lambda row: getattr(row, 'microsecond') // 1000


@dispatch(DateTimeTruncate)
def rowfunc(expr):
    return partial(pydatetime.truncate, measure=expr.measure, unit=expr.unit)


@dispatch(UnaryMath)
def rowfunc(expr):
    return getattr(math, type(expr).__name__)


@dispatch(USub)
def rowfunc(expr):
    return operator.neg


@dispatch(Not)
def rowfunc(expr):
    return operator.invert


@dispatch(Arithmetic)
def rowfunc(expr):
    if not isinstance(expr.lhs, Expr):
        return lambda x: expr.op(expr.lhs, x)
    if not isinstance(expr.rhs, Expr):
        return lambda x: expr.op(x, expr.rhs)
    return expr.op


@dispatch(ElemWise, base)
def compute_up(expr, data, **kwargs):
    return rowfunc(expr)(data)


@dispatch(notnull)
def rowfunc(_):
    return lambda x: x is not None


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


def deepmap(func, *data, **kwargs):
    """

    >>> inc = lambda x: x + 1
    >>> list(deepmap(inc, [1, 2], n=1))
    [2, 3]
    >>> list(deepmap(inc, [(1, 2), (3, 4)], n=2))
    [(2, 3), (4, 5)]

    Works on variadic args too

    >>> add = lambda x, y: x + y
    >>> list(deepmap(add, [1, 2], [10, 20], n=1))
    [11, 22]
    """
    n = kwargs.pop('n', 1)
    if n == 0:
        return func(*data)
    if n == 1:
        return map(func, *data)
    else:
        return map(compose(tuple, partial(deepmap, func, n=n - 1)), *data)


@dispatch(Merge)
def rowfunc(t):
    children = [optimize(child, []) for child in t.children]
    funcs = [rrowfunc(_child, t._child) for _child in children]
    return compose(concat_maybe_tuples, juxt(*funcs))


@dispatch(ElemWise, Sequence)
def compute_up(t, seq, **kwargs):
    func = rowfunc(t)
    if iscollection(t._child.dshape):
        return deepmap(func, seq, n=ndim(child(t)))
    else:
        return func(seq)


@dispatch(Broadcast, Sequence)
def compute_up(t, seq, **kwargs):
    func = rowfunc(t)
    return deepmap(func, seq, n=ndim(child(t)))


@dispatch(Arithmetic, Sequence + (int, float, bool))
def compute_up(t, seq, **kwargs):
    func = rowfunc(t)
    return deepmap(func, seq, n=ndim(child(t)))


@dispatch(Arithmetic, Sequence, Sequence)
def compute_up(t, a, b, **kwargs):
    if ndim(t.lhs) != ndim(t.rhs):
        raise ValueError()
    # TODO: Tee if necessary
    func = rowfunc(t)
    return deepmap(func, a, b, n=ndim(t.lhs))


@dispatch(Selection, Sequence)
def compute_up(t, seq, **kwargs):
    predicate = optimize(t.predicate, seq)
    predicate = rrowfunc(predicate, child(t))
    return filter(predicate, seq)


@dispatch(Selection, Sequence, Sequence)
def compute_up(expr, seq, predicate, **kwargs):
    preds = iter(predicate)
    return filter(lambda _: next(preds), seq)


@dispatch(Reduction, Sequence)
def compute_up(t, seq, **kwargs):
    if t.axis != (0,):
        raise NotImplementedError('Only 1D reductions currently supported')
    result = compute_up_1d(t, seq, **kwargs)
    if t.keepdims:
        return (result,)
    else:
        return result


@dispatch(Reduction, Sequence)
def compute_up_1d(t, seq, **kwargs):
    op = getattr(builtins, t.symbol)
    return op(seq)


@dispatch(nelements, Sequence)
def compute_up_1d(expr, seq, **kwargs):
    try:
        return len(seq)
    except TypeError:
        return toolz.count(seq)


@dispatch(ElemWise, base)
def compute_up(expr, data, **kwargs):
    return rowfunc(expr)(data)


@dispatch(BinOp, base, base)
def compute_up(bop, a, b, **kwargs):
    return bop.op(a, b)


@dispatch(UnaryOp, base)
def compute_up(uop, x, **kwargs):
    return uop.op(x)


@dispatch(UnaryMath, numbers.Real)
def compute_up(f, n, **kwargs):
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
def compute_up_1d(t, seq, **kwargs):
    return toolz.count(filter(None, seq))


@dispatch(Distinct, Sequence)
def compute_up(t, seq, **kwargs):
    if t.on:
        raise NotImplementedError(
            'python backend cannot specify what columns to distinct on'
        )
    try:
        row = toolz.first(seq)
    except StopIteration:
        return ()
    seq = concat([[row], seq])  # re-add row to seq

    if isinstance(row, list):
        seq = map(tuple, seq)

    return unique(seq)


@dispatch(nunique, Sequence)
def compute_up_1d(t, seq, **kwargs):
    return len(set(seq))


@dispatch(mean, Sequence)
def compute_up_1d(t, seq, **kwargs):
    return _mean(seq)


@dispatch(var, Sequence)
def compute_up_1d(t, seq, **kwargs):
    return _var(seq, t.unbiased)


@dispatch(std, Sequence)
def compute_up_1d(t, seq, **kwargs):
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
binops = {reductions.sum: (add, add, 0),
          reductions.min: (lesser, lesser, 1e250),
          reductions.max: (greater, lesser, -1e250),
          reductions.count: (countit, add, 0),
          reductions.any: (or_, or_, False),
          reductions.all: (and_, and_, True)}


def child(expr):
    if len(expr._inputs) > 1:
        raise ValueError()
    return expr._inputs[0]


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
        compute_up(By, Sequence)
    """
    grouper = rrowfunc(t.grouper, t._child)
    if (isinstance(t.apply, Reduction) and
            type(t.apply) in binops):

        binop, combiner, initial = binops[type(t.apply)]
        applier = rrowfunc(t.apply._child, t._child)

        def binop2(acc, x):
            return binop(acc, applier(x))

        return grouper, binop2, combiner, initial

    elif (isinstance(t.apply, Summary) and
          builtins.all(type(val) in binops for val in t.apply.values)):

        binops2, combiners, inits = zip(
            *[binops[type(v)] for v in t.apply.values])
        appliers = [rrowfunc(v._child, t._child) for v in t.apply.values]

        def binop2(accs, x):
            return tuple(binop(acc, applier(x)) for binop, acc, applier in
                         zip(binops2, accs, appliers))

        def combiner(a, b):
            return tuple(c(x, y) for c, x, y in zip(combiners, a, b))

        return grouper, binop2, combiner, tuple(inits)


@dispatch(By, Sequence)
def compute_up(t, seq, **kwargs):
    apply = optimize(t.apply, seq)
    grouper = optimize(t.grouper, seq)
    t = By(grouper, apply)
    if ((isinstance(t.apply, Reduction) and type(t.apply) in binops) or
        (isinstance(t.apply, Summary) and builtins.all(type(val) in binops
                                                       for val in t.apply.values))):
        grouper, binop, combiner, initial = reduce_by_funcs(t)
        d = reduceby(grouper, binop, seq, initial)
    else:
        grouper = rrowfunc(t.grouper, t._child)
        groups = groupby(grouper, seq)
        d = dict((k, compute(t.apply, {t._child: v}, return_type='native'))
                 for k, v in groups.items())

    if isscalar(t.grouper.dshape.measure):
        keyfunc = lambda x: (x,)
    else:
        keyfunc = identity
    if isscalar(t.apply.dshape.measure):
        valfunc = lambda x: (x,)
    else:
        valfunc = identity
    return tuple(keyfunc(k) + valfunc(v) for k, v in d.items())


def pair_assemble(t, on_left=None, on_right=None):
    """ Combine a pair of records into a single record

    This is mindful to shared columns as well as missing records

    Parameters
    ----------
    t : Join
        The join to combine.
    on_left : list of column indicies, optional
        The column indicies of the left columns.
    on_right : list of column indicies, optional
        The column indicies of the right columns.

    Returns
    -------
    assemble : callable
        A function that assembles the data from a row of the
        left and right data sources.
    """
    try:
        from cytoolz import get  # not curried version
    except:
        from toolz import get
    on_left = (
        on_left
        if on_left is not None else
        [t.lhs.fields.index(col) for col in listpack(t.on_left)]
    )
    on_right = (
        on_right
        if on_right is not None else
        [t.rhs.fields.index(col) for col in listpack(t.on_right)]
    )

    left_self_columns = [t.lhs.fields.index(c) for c in t.lhs.fields
                         if c not in listpack(t.on_left)]
    right_self_columns = [t.rhs.fields.index(c) for c in t.rhs.fields
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
            left_entries = (None,) * (len(t.lhs.fields) - len(on_left))

        if b is not None:
            right_entries = get(right_self_columns, b)
        else:
            right_entries = (None,) * (len(t.rhs.fields) - len(on_right))

        return joined + left_entries + right_entries

    return assemble


@dispatch(Join, Sequence, Sequence)
def compute_up(t, lhs, rhs, **kwargs):
    """ Join Operation for Python Streaming Backend

    Note that a pure streaming Join is challenging/impossible because any row
    in one seq might connect to any row in the other, requiring simultaneous
    complete access.

    As a result this approach compromises and fully realizes the LEFT sequence
    while allowing the RIGHT sequence to stream.  As a result

    Always put your bigger collection on the RIGHT side of the Join.
    """
    if lhs == rhs:
        lhs, rhs = itertools.tee(lhs, 2)

    on_left = [t.lhs.fields.index(col) for col in listpack(t.on_left)]
    on_right = [t.rhs.fields.index(col) for col in listpack(t.on_right)]

    left_default = (None if t.how in ('right', 'outer')
                    else toolz.itertoolz.no_default)
    right_default = (None if t.how in ('left', 'outer')
                     else toolz.itertoolz.no_default)

    pairs = toolz.join(on_left, lhs,
                       on_right, rhs,
                       left_default=left_default,
                       right_default=right_default)

    assemble = pair_assemble(t, on_left, on_right)

    return map(assemble, pairs)


@dispatch(Sort, Sequence)
def compute_up(t, seq, **kwargs):
    if isscalar(t._child.dshape.measure) and t.key == t._child._name:
        key = identity
    elif isinstance(t.key, (str, unicode, tuple, list)):
        key = rowfunc(t._child[t.key])
    else:
        key = rrowfunc(optimize(t.key, seq), t._child)
    return sorted(seq,
                  key=key,
                  reverse=not t.ascending)


@dispatch(Head, Sequence)
def compute_up(t, seq, **kwargs):
    if t.n < 100:
        return tuple(take(t.n, seq))
    else:
        return take(t.n, seq)


@dispatch(Sample, Sequence)
def compute_up(t, seq, **kwargs):
    nsamp = t.n if t.n is not None else int(t.frac * len(seq))
    return random.sample(seq, min(nsamp, len(seq)))


@dispatch((Label, ReLabel), Sequence)
def compute_up(t, seq, **kwargs):
    return seq


@dispatch(Apply, Sequence)
def compute_up(t, seq, **kwargs):
    return t.func(seq)


@dispatch(Summary, Sequence)
def compute_up(expr, data, **kwargs):
    if expr._child.ndim != 1:
        raise NotImplementedError('Only 1D reductions currently supported')
    if isinstance(data, Iterator):
        datas = itertools.tee(data, len(expr.values))
        result = tuple(
            compute(val, {expr._child: data}, return_type='native')
            for val, data in zip(expr.values, datas)
        )
    else:
        result = tuple(
            compute(val, {expr._child: data}, return_type='native')
            for val in expr.values
        )

    if expr.keepdims:
        return (result,)
    else:
        return result


@dispatch(Like, Sequence)
def compute_up(expr, seq, **kwargs):
    def func(x, pattern=expr.pattern):
        return fnmatch.fnmatch(x, pattern)
    return map(func, seq)


@dispatch(Slice, Sequence)
def compute_up(expr, seq, **kwargs):
    index = expr.index
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, _inttypes):
        try:
            return seq[index]
        except:
            if index >= 0:
                return nth(index, seq)
            else:
                return tail(-index, seq)[0]
    if isinstance(index, slice):
        if (index.start and index.start < 0 and
                index.stop is None and
                index.step in (1, None)):
            return tail(-index.start, seq)
        else:
            return itertools.islice(seq, index.start, index.stop, index.step)
    raise NotImplementedError("Only 1d slices supported")


@dispatch(Coerce, (np.float32, np.float64, np.int64, np.int32, base))
def compute_up(expr, ob, **kwargs):
    tp = expr.to
    shape = tp.shape
    if shape:
        raise TypeError(
            'cannot convert scalar object %r to array or matrix shape %r' % (
                ob,
                shape,
            ),
        )

    measure = tp.measure
    if isinstance(measure, Option):
        if pd.isnull(ob):
            return None
        measure = measure.ty
    dtype = to_numpy_dtype(measure)
    return dtype.type(ob)
