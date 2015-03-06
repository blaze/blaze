"""

>>> from blaze.expr import symbol
>>> from blaze.compute.pandas import compute

>>> accounts = symbol('accounts', 'var * {name: string, amount: int}')
>>> deadbeats = accounts[accounts['amount'] < 0]['name']

>>> from pandas import DataFrame
>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> df = DataFrame(data, columns=['name', 'amount'])
>>> compute(deadbeats, df)
1        Bob
2    Charlie
Name: name, dtype: object
"""
from __future__ import absolute_import, division, print_function

import pandas as pd
from pandas.core.generic import NDFrame
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import numpy as np
from toolz import merge as merge_dicts
from toolz.curried import pipe, filter, map, concat
import fnmatch
from datashape.predicates import isscalar
import datashape
import itertools

from odo import into
from ..dispatch import dispatch
from ..expr import (Projection, Field, Sort, Head, Broadcast, Selection,
                    Reduction, Distinct, Join, By, Summary, Label, ReLabel,
                    Map, Apply, Merge, std, var, Like, Slice, summary,
                    ElemWise, DateTime, Millisecond, Expr, Symbol,
                    UTCFromTimestamp, nelements, DateTimeTruncate, count)
from ..expr import UnaryOp, BinOp
from ..expr import symbol, common_subexpression
from .core import compute, compute_up, base
from ..compatibility import _inttypes

__all__ = []


@dispatch(Projection, DataFrame)
def compute_up(t, df, **kwargs):
    return df[list(t.fields)]


@dispatch(Field, (DataFrame, DataFrameGroupBy))
def compute_up(t, df, **kwargs):
    assert len(t.fields) == 1
    return df[t.fields[0]]


@dispatch(Field, Series)
def compute_up(t, data, **kwargs):
    assert len(t.fields) == 1
    if t.fields[0] == data.name:
        return data
    else:
        raise ValueError("Fieldname %r does not match Series name %r"
                         % (t.fields[0], data.name))


@dispatch(Broadcast, DataFrame)
def compute_up(t, df, **kwargs):
    d = dict((t._child[c]._expr, df[c]) for c in t._child.fields)
    return compute(t._expr, d)


@dispatch(Broadcast, Series)
def compute_up(t, s, **kwargs):
    return compute_up(t, s.to_frame(), **kwargs)


@dispatch(BinOp, Series)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)


@dispatch(BinOp, Series, (Series, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (Series, base), Series)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, NDFrame)
def compute_up(t, df, **kwargs):
    f = getattr(t, 'op', getattr(np, t.symbol, None))
    if f is None:
        raise ValueError('%s is not a valid operation on %s objects' %
                         (t.symbol, type(df).__name__))
    return f(df)


@dispatch(Selection, (Series, DataFrame))
def compute_up(t, df, **kwargs):
    predicate = compute(t.predicate, {t._child: df})
    return df[predicate]


@dispatch(Join, DataFrame, DataFrame)
def compute_up(t, lhs, rhs, **kwargs):
    """ Join two pandas data frames on arbitrary columns

    The approach taken here could probably be improved.

    To join on two columns we force each column to be the index of the
    dataframe, perform the join, and then reset the index back to the left
    side's original index.
    """
    result = pd.merge(lhs, rhs,
                      left_on=t.on_left, right_on=t.on_right,
                      how=t.how)
    return result.reset_index()[t.fields]


@dispatch(Symbol, (DataFrameGroupBy, SeriesGroupBy))
def compute_up(t, gb, **kwargs):
    return gb


def get_scalar(result):
    # pandas may return an int, numpy scalar or non scalar here so we need to
    # program defensively so that things are JSON serializable
    try:
        return result.item()
    except (AttributeError, ValueError):
        return result


@dispatch(Reduction, (Series, SeriesGroupBy))
def compute_up(t, s, **kwargs):
    result = get_scalar(getattr(s, t.symbol)())
    if t.keepdims:
        result = Series([result], name=s.name)
    return result


@dispatch((std, var), (Series, SeriesGroupBy))
def compute_up(t, s, **kwargs):
    result = get_scalar(getattr(s, t.symbol)(ddof=t.unbiased))
    if t.keepdims:
        result = Series([result], name=s.name)
    return result


@dispatch(Distinct, (DataFrame, Series))
def compute_up(t, df, **kwargs):
    return df.drop_duplicates().reset_index(drop=True)


def unpack(seq):
    """ Unpack sequence of length one

    >>> unpack([1, 2, 3])
    [1, 2, 3]

    >>> unpack([1])
    1
    """
    seq = list(seq)
    if len(seq) == 1:
        seq = seq[0]
    return seq


Grouper = ElemWise, Series, list


@dispatch(By, list, DataFrame)
def get_grouper(c, grouper, df):
    return grouper


@dispatch(By, Expr, NDFrame)
def get_grouper(c, grouper, df):
    g = compute(grouper, {c._child: df})
    if isinstance(g, Series):
        return g
    if isinstance(g, DataFrame):
        return [g[col] for col in g.columns]


@dispatch(By, (Field, Projection), NDFrame)
def get_grouper(c, grouper, df):
    return grouper.fields


@dispatch(By, Reduction, Grouper, NDFrame)
def compute_by(t, r, g, df):
    names = [r._name]
    preapply = compute(r._child, {t._child: df})

    # Pandas and Blaze column naming schemes differ
    # Coerce DataFrame column names to match Blaze's names
    preapply = preapply.copy()
    if isinstance(preapply, Series):
        preapply.name = names[0]
    else:
        preapply.names = names
    group_df = concat_nodup(df, preapply)

    gb = group_df.groupby(g)
    groups = gb[names[0] if isscalar(t.apply._child.dshape.measure) else names]

    return compute_up(r, groups)  # do reduction


name_dict = dict()
seen_names = set()


def _name(expr):
    """ A unique and deterministic name for an expression """
    if expr in name_dict:
        return name_dict[expr]
    result = base = expr._name or '_'
    if result in seen_names:
        for i in itertools.count(1):
            result = '%s_%d' % (base, i)
            if result not in seen_names:
                break
    # result is an unseen name
    seen_names.add(result)
    name_dict[expr] = result
    return result


def fancify_summary(expr):
    """ Separate a complex summary into two pieces

    Helps pandas compute_by on summaries

    >>> t = symbol('t', 'var * {x: int, y: int}')
    >>> one, two, three = fancify_summary(summary(a=t.x.sum(), b=t.x.sum() + t.y.count() - 1))

    A simpler summary with only raw reductions
    >>> one
    summary(x_sum=sum(t.x), y_count=count(t.y))

    A mapping of those names to new leaves to use in another compuation
    >>> two  # doctest: +SKIP
    {'x_sum': x_sum, 'y_count': y_count}

    A mapping of computations to do for each column
    >>> three   # doctest: +SKIP
    {'a': x_sum, 'b': (x_sum + y_count) - 1}

    In this way, ``compute_by`` is able to do simple pandas reductions using
    groups.agg(...) and then do columnwise arithmetic afterwards.
    """
    seen_names.clear()
    name_dict.clear()
    exprs = pipe(expr.values,
                 map(Expr._traverse),
                 concat,
                 filter(lambda x: isinstance(x, Reduction)),
                 set)
    one = summary(**dict((_name(expr), expr) for expr in exprs))

    two = dict((_name(expr), symbol(_name(expr), datashape.var * expr.dshape))
               for expr in exprs)

    d = dict((expr, two[_name(expr)]) for expr in exprs)
    three = dict((name, value._subs(d)) for name, value in zip(expr.names,
                                                               expr.values))

    return one, two, three


@dispatch(By, Summary, Grouper, NDFrame)
def compute_by(t, s, g, df):
    one, two, three = fancify_summary(s)  # see above
    names = one.fields
    preapply = DataFrame(dict(zip(names,
                                  [compute(v._child, {t._child: df})
                                   for v in one.values])))

    df2 = concat_nodup(df, preapply)

    groups = df2.groupby(g)

    d = dict((name, v.symbol) for name, v in zip(one.names, one.values))

    result = groups.agg(d)

    scope = dict((v, result[k]) for k, v in two.items())
    cols = [compute(expr.label(name), scope) for name, expr in three.items()]

    result2 = pd.concat(cols, axis=1)

    # Rearrange columns to match names order
    result3 = result2[sorted(result2.columns, key=lambda t: s.fields.index(t))]
    return result3


@dispatch(Expr, DataFrame)
def post_compute_by(t, df):
    return df.reset_index(drop=True)


@dispatch((Summary, Reduction), DataFrame)
def post_compute_by(t, df):
    return df.reset_index()


@dispatch(By, NDFrame)
def compute_up(t, df, **kwargs):
    grouper = get_grouper(t, t.grouper, df)
    result = compute_by(t, t.apply, grouper, df)
    result2 = post_compute_by(t.apply, into(DataFrame, result))
    if isinstance(result2, DataFrame):
        result2.columns = t.fields
    return result2


def concat_nodup(a, b):
    """ Concatenate two dataframes/series without duplicately named columns


    >>> df = DataFrame([[1, 'Alice',   100],
    ...                 [2, 'Bob',    -200],
    ...                 [3, 'Charlie', 300]],
    ...                columns=['id','name', 'amount'])

    >>> concat_nodup(df, df)
       id     name  amount
    0   1    Alice     100
    1   2      Bob    -200
    2   3  Charlie     300


    >>> concat_nodup(df.name, df.amount)
          name  amount
    0    Alice     100
    1      Bob    -200
    2  Charlie     300



    >>> concat_nodup(df, df.amount + df.id)
       id     name  amount    0
    0   1    Alice     100  101
    1   2      Bob    -200 -198
    2   3  Charlie     300  303
    """

    if isinstance(a, DataFrame) and isinstance(b, DataFrame):
        return pd.concat([a, b[[c for c in b.columns if c not in a.columns]]],
                         axis=1)
    if isinstance(a, DataFrame) and isinstance(b, Series):
        if b.name not in a.columns:
            return pd.concat([a, b], axis=1)
        else:
            return a
    if isinstance(a, Series) and isinstance(b, DataFrame):
        return pd.concat([a, b[[c for c in b.columns if c != a.name]]], axis=1)
    if isinstance(a, Series) and isinstance(b, Series):
        if a.name == b.name:
            return a
        else:
            return pd.concat([a, b], axis=1)


@dispatch(Sort, DataFrame)
def compute_up(t, df, **kwargs):
    return df.sort(t.key, ascending=t.ascending)


@dispatch(Sort, Series)
def compute_up(t, s, **kwargs):
    return s.order(ascending=t.ascending)


@dispatch(Head, (Series, DataFrame))
def compute_up(t, df, **kwargs):
    return df.head(t.n)


@dispatch(Label, DataFrame)
def compute_up(t, df, **kwargs):
    return DataFrame(df, columns=[t.label])


@dispatch(Label, Series)
def compute_up(t, df, **kwargs):
    return Series(df, name=t.label)


@dispatch(ReLabel, DataFrame)
def compute_up(t, df, **kwargs):
    return df.rename(columns=dict(t.labels))


@dispatch(ReLabel, Series)
def compute_up(t, s, **kwargs):
    labels = t.labels
    if len(labels) > 1:
        raise ValueError('You can only relabel a Series with a single name')
    pair, = labels
    _, replacement = pair
    return Series(s, name=replacement)


@dispatch(Map, DataFrame)
def compute_up(t, df, **kwargs):
    return df.apply(lambda tup: t.func(*tup), axis=1)


@dispatch(Map, Series)
def compute_up(t, df, **kwargs):
    result = df.map(t.func)
    try:
        result.name = t._name
    except NotImplementedError:
        # We don't have a schema, but we should still be able to map
        result.name = df.name
    return result


@dispatch(Apply, (Series, DataFrame))
def compute_up(t, df, **kwargs):
    return t.func(df)


@dispatch(Merge, NDFrame)
def compute_up(t, df, scope=None, **kwargs):
    subexpression = common_subexpression(*t.children)
    scope = merge_dicts(scope or {}, {subexpression: df})
    children = [compute(_child, scope) for _child in t.children]
    return pd.concat(children, axis=1)


@dispatch(Summary, DataFrame)
def compute_up(expr, data, **kwargs):
    values = [compute(val, {expr._child: data}) for val in expr.values]
    if expr.keepdims:
        return DataFrame([values], columns=expr.fields)
    else:
        return Series(dict(zip(expr.fields, values)))


@dispatch(Summary, Series)
def compute_up(expr, data, **kwargs):
    result = tuple(compute(val, {expr._child: data}) for val in expr.values)
    if expr.keepdims:
        result = [result]
    return result


@dispatch(Like, DataFrame)
def compute_up(expr, df, **kwargs):
    arrs = [df[name].str.contains('^%s$' % fnmatch.translate(pattern))
            for name, pattern in expr.patterns.items()]
    return df[np.logical_and.reduce(arrs)]


def get_date_attr(s, attr):
    try:
        # new in pandas 0.15
        return getattr(s.dt, attr)
    except AttributeError:
        return getattr(pd.DatetimeIndex(s), attr)


@dispatch(DateTime, Series)
def compute_up(expr, s, **kwargs):
    return get_date_attr(s, expr.attr)


@dispatch(UTCFromTimestamp, Series)
def compute_up(expr, s, **kwargs):
    return pd.datetools.to_datetime(s * 1e9, utc=True)


@dispatch(Millisecond, Series)
def compute_up(_, s, **kwargs):
    return get_date_attr(s, 'microsecond') // 1000


@dispatch(Slice, (DataFrame, Series))
def compute_up(expr, df, **kwargs):
    index = expr.index
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, _inttypes + (list,)):
        return df.iloc[index]
    elif isinstance(index, slice):
        if index.stop is not None:
            return df.iloc[index.start:index.stop:index.step]
        else:
            return df.iloc[index]
    else:
        raise NotImplementedError()


@dispatch(count, DataFrame)
def compute_up(expr, df, **kwargs):
    result = df.shape[0]
    if expr.keepdims:
        result = Series([result])
    return result


@dispatch(nelements, (DataFrame, Series))
def compute_up(expr, df, **kwargs):
    return df.shape[0]


units_map = {
    'year': 'Y',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'hour': 'h',
    'minute': 'm',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us',
    'nanosecond': 'ns'
}


@dispatch(DateTimeTruncate, Series)
def compute_up(expr, data, **kwargs):
    return Series(compute_up(expr, into(np.ndarray, data), **kwargs))
