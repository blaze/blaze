"""

>>> from blaze.expr import Symbol
>>> from blaze.compute.pandas import compute

>>> accounts = Symbol('accounts', 'var * {name: string, amount: int}')
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
from collections import defaultdict
from toolz import merge as merge_dicts
import fnmatch
from datashape.predicates import isscalar

from ..api.into import into
from ..dispatch import dispatch
from ..expr import (Projection, Field, Sort, Head, Broadcast, Selection,
                    Reduction, Distinct, Join, By, Summary, Label, ReLabel,
                    Map, Apply, Merge, Union, std, var, Like, Slice,
                    ElemWise, DateTime, Millisecond, Expr, Symbol,
                    UTCFromTimestamp)
from ..expr import UnaryOp, BinOp
from ..expr import Symbol, common_subexpression
from .core import compute, compute_up, base
from ..compatibility import _inttypes

__all__ = []


@dispatch(Projection, DataFrame)
def compute_up(t, df, **kwargs):
    return df[list(t.fields)]


@dispatch(Field, (DataFrame, DataFrameGroupBy))
def compute_up(t, df, **kwargs):
    return df[t.fields[0]]


@dispatch(Field, Series)
def compute_up(t, data, **kwargs):
    if t.fields[0] == data.name:
        return data
    else:
        raise ValueError("Fieldname %s does not match Series name %s"
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


@dispatch(Symbol, DataFrame)
def compute_up(t, df, **kwargs):
    if not list(t.fields) == list(df.names):
        # TODO also check dtype
        raise ValueError("Schema mismatch: \n\nTable:\n%s\n\nDataFrame:\n%s"
                         % (t, df))
    return df


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


def post_reduction(result):
    # pandas may return an int, numpy scalar or non scalar here so we need to
    # program defensively so that things are JSON serializable
    try:
        return result.item()
    except (AttributeError, ValueError):
        return result


@dispatch(Reduction, (Series, SeriesGroupBy))
def compute_up(t, s, **kwargs):
    result = post_reduction(getattr(s, t.symbol)())
    if t.keepdims:
        result = Series([result], name=s.name)
    return result


@dispatch((std, var), (Series, SeriesGroupBy))
def compute_up(t, s, **kwargs):
    result = post_reduction(getattr(s, t.symbol)(ddof=t.unbiased))
    if t.keepdims:
        result = Series([result], name=s.name)
    return result


@dispatch(Distinct, DataFrame)
def compute_up(t, df, **kwargs):
    return df.drop_duplicates()


@dispatch(Distinct, Series)
def compute_up(t, s, **kwargs):
    s2 = Series(s.unique())
    s2.name = s.name
    return s2


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


@dispatch(By, (ElemWise, Series), NDFrame)
def get_grouper(c, grouper, df):
    return compute(grouper, {c._child: df})


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


@dispatch(By, Summary, Grouper, NDFrame)
def compute_by(t, s, g, df):
    names = s.fields
    preapply = DataFrame(dict(zip(names,
                                  (compute(v._child, {t._child: df})
                                   for v in s.values))))

    df2 = concat_nodup(df, preapply)

    groups = df2.groupby(g)

    d = defaultdict(list)
    for name, v in zip(names, s.values):
        d[name].append(getattr(Series, v.symbol))

    result = groups.agg(dict(d))

    # Rearrange columns to match names order
    result = result[sorted(result.columns, key=lambda t: names.index(t[0]))]
    result.columns = t.apply.fields  # flatten down multiindex
    return result


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
    return post_compute_by(t.apply, into(DataFrame, result))


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


@dispatch(Union, (Series, DataFrame), tuple)
def compute_up(t, example, children, **kwargs):
    return pd.concat(children, axis=0)


@dispatch(Summary, DataFrame)
def compute_up(expr, data, **kwargs):
    values = [compute(val, {expr._child: data}) for val in expr.values]
    if expr.keepdims:
        return DataFrame([values], columns=expr.fields)
    else:
        return Series(dict(zip(expr.fields, values)))


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
    return pd.datetools.to_datetime(s*1e9, utc=True)


@dispatch(Millisecond, Series)
def compute_up(_, s, **kwargs):
    return get_date_attr(s, 'microsecond') // 1000


@dispatch(Slice, (DataFrame, Series))
def compute_up(expr, df, **kwargs):
    index = expr.index
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if isinstance(index, _inttypes):
        return df.iloc[index]
    elif isinstance(index, slice):
        if index.stop is not None:
            return df.iloc[slice(index.start,
                                index.stop,
                                index.step)]
        else:
            return df.iloc[index]
    else:
        raise NotImplementedError()
