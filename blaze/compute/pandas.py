"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.pandas import compute

>>> accounts = TableSymbol('accounts', '{name: string, amount: int}')
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

from ..api.into import into
from ..dispatch import dispatch
from ..expr import (Projection, Column, Sort, Head, ColumnWise, Selection,
                    Reduction, Distinct, Join, By, Summary, Label, ReLabel,
                    Map, Apply, Merge, Union, TableExpr, std, var, Like)
from ..expr import UnaryOp, BinOp
from ..expr import TableSymbol, common_subexpression
from .core import compute, compute_one, base

__all__ = []


@dispatch(Projection, DataFrame)
def compute_one(t, df, **kwargs):
    return df[list(t.columns)]


@dispatch(Column, (DataFrame, DataFrameGroupBy))
def compute_one(t, df, **kwargs):
    return df[t.columns[0]]


@dispatch(Column, (Series, SeriesGroupBy))
def compute_one(_, s, **kwargs):
    return s


@dispatch(ColumnWise, DataFrame)
def compute_one(t, df, **kwargs):
    d = dict((t.child[c].scalar_symbol, df[c]) for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(ColumnWise, Series)
def compute_one(t, s, **kwargs):
    return compute_one(t, s.to_frame(), **kwargs)


@dispatch(BinOp, Series, (Series, base))
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (Series, base), Series)
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, NDFrame)
def compute_one(t, df, **kwargs):
    f = getattr(t, 'op', getattr(np, t.symbol, None))
    if f is None:
        raise ValueError('%s is not a valid operation on %s objects' %
                         (t.symbol, type(df).__name__))
    return f(df)


@dispatch(Selection, (Series, DataFrame))
def compute_one(t, df, **kwargs):
    predicate = compute(t.predicate, {t.child: df})
    return df[predicate]


@dispatch(TableSymbol, DataFrame)
def compute_one(t, df, **kwargs):
    if not list(t.columns) == list(df.columns):
        # TODO also check dtype
        raise ValueError("Schema mismatch: \n\nTable:\n%s\n\nDataFrame:\n%s"
                         % (t, df))
    return df


@dispatch(Join, DataFrame, DataFrame)
def compute_one(t, lhs, rhs, **kwargs):
    """ Join two pandas data frames on arbitrary columns

    The approach taken here could probably be improved.

    To join on two columns we force each column to be the index of the
    dataframe, perform the join, and then reset the index back to the left
    side's original index.
    """
    result = pd.merge(lhs, rhs,
                      left_on=t.on_left, right_on=t.on_right,
                      how=t.how)
    return result.reset_index()[t.columns]


@dispatch(TableSymbol, (DataFrameGroupBy, SeriesGroupBy))
def compute_one(t, gb, **kwargs):
    return gb


@dispatch(Reduction, (DataFrame, DataFrameGroupBy))
def compute_one(t, df, **kwargs):
    return getattr(df, t.symbol)()


@dispatch((std, var), (DataFrame, DataFrameGroupBy))
def compute_one(t, df, **kwargs):
    return getattr(df, t.symbol)(ddof=t.unbiased)


def post_reduction(result):
    # pandas may return an int, numpy scalar or non scalar here so we need to
    # program defensively so that things are JSON serializable
    try:
        return result.item()
    except (AttributeError, ValueError):
        return result


@dispatch(Reduction, (Series, SeriesGroupBy))
def compute_one(t, s, **kwargs):
    return post_reduction(getattr(s, t.symbol)())


@dispatch((std, var), (Series, SeriesGroupBy))
def compute_one(t, s, **kwargs):
    return post_reduction(getattr(s, t.symbol)(ddof=t.unbiased))


@dispatch(Distinct, DataFrame)
def compute_one(t, df, **kwargs):
    return df.drop_duplicates()


@dispatch(Distinct, Series)
def compute_one(t, s, **kwargs):
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


Grouper = Column, ColumnWise, Series, list


@dispatch(By, list, DataFrame)
def get_grouper(c, grouper, df):
    return grouper


@dispatch(By, (Column, ColumnWise, Series), NDFrame)
def get_grouper(c, grouper, df):
    return compute(grouper, {c.child: df})


@dispatch(By, Projection, NDFrame)
def get_grouper(c, grouper, df):
    return grouper.columns


@dispatch(By, Reduction, Grouper, NDFrame)
def compute_by(t, r, g, df):
    names = r.dshape[0].names
    preapply = compute(r.child, {t.child: df})

    # Pandas and Blaze column naming schemes differ
    # Coerce DataFrame column names to match Blaze's names
    preapply = preapply.copy()
    if isinstance(preapply, Series):
        preapply.name = names[0]
    else:
        preapply.columns = names
    group_df = concat_nodup(df, preapply)

    gb = group_df.groupby(g)
    groups = gb[names[0] if t.apply.child.iscolumn else names]

    return compute_one(r, groups)  # do reduction


@dispatch(By, Summary, Grouper, NDFrame)
def compute_by(t, s, g, df):
    names = s.names
    preapply = DataFrame(dict(zip(names,
                                  (compute(v.child, {t.child: df})
                                   for v in s.values))))

    df2 = concat_nodup(df, preapply)

    groups = df2.groupby(g)

    d = defaultdict(list)
    for name, v in zip(names, s.values):
        d[name].append(getattr(Series, v.symbol))

    result = groups.agg(dict(d))

    # Rearrange columns to match names order
    result = result[sorted(result.columns, key=lambda t: names.index(t[0]))]
    result.columns = t.apply.names  # flatten down multiindex
    return result


@dispatch(TableExpr, DataFrame)
def post_compute_by(t, df):
    return df.reset_index(drop=True)


@dispatch((Summary, Reduction), DataFrame)
def post_compute_by(t, df):
    return df.reset_index()


@dispatch(By, NDFrame)
def compute_one(t, df, **kwargs):
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
def compute_one(t, df, **kwargs):
    return df.sort(t.key, ascending=t.ascending)


@dispatch(Sort, Series)
def compute_one(t, s, **kwargs):
    return s.order(ascending=t.ascending)


@dispatch(Head, (Series, DataFrame))
def compute_one(t, df, **kwargs):
    return df.head(t.n)


@dispatch(Label, DataFrame)
def compute_one(t, df, **kwargs):
    return DataFrame(df, columns=[t.label])


@dispatch(Label, Series)
def compute_one(t, df, **kwargs):
    return Series(df, name=t.label)


@dispatch(ReLabel, DataFrame)
def compute_one(t, df, **kwargs):
    return df.rename(columns=dict(t.labels))


@dispatch(ReLabel, Series)
def compute_one(t, s, **kwargs):
    labels = t.labels
    if len(labels) > 1:
        raise ValueError('You can only relabel a Series with a single name')
    pair, = labels
    _, replacement = pair
    return Series(s, name=replacement)


@dispatch(Map, DataFrame)
def compute_one(t, df, **kwargs):
    return df.apply(lambda tup: t.func(*tup), axis=1)


@dispatch(Map, Series)
def compute_one(t, df, **kwargs):
    result = df.map(t.func)
    try:
        result.name = t.name
    except NotImplementedError:
        # We don't have a schema, but we should still be able to map
        result.name = df.name
    return result


@dispatch(Apply, (Series, DataFrame))
def compute_one(t, df, **kwargs):
    return t.func(df)


@dispatch(Merge, NDFrame)
def compute_one(t, df, scope=None, **kwargs):
    subexpression = common_subexpression(*t.children)
    scope = merge_dicts(scope or {}, {subexpression: df})
    children = [compute(child, scope) for child in t.children]
    return pd.concat(children, axis=1)


@dispatch(Union, DataFrame, tuple)
def compute_one(t, example, children, **kwargs):
    return pd.concat(children, axis=0)


@dispatch(Summary, DataFrame)
def compute_one(expr, data, **kwargs):
    return Series(dict(zip(expr.names, [compute(val, {expr.child: data})
                                        for val in expr.values])))


@dispatch(Like, DataFrame)
def compute_one(expr, df, **kwargs):
    arrs = [df[name].str.contains('^%s$' % fnmatch.translate(pattern))
            for name, pattern in expr.patterns.items()]
    return df[np.logical_and.reduce(arrs)]
