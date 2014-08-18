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
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import numpy as np
from collections import defaultdict

from ..dispatch import dispatch
from ..expr import *
from .core import compute, compute_one, base

__all__ = []


@dispatch(Projection, DataFrame)
def compute_one(t, df, **kwargs):
    return df[list(t.columns)]


@dispatch(Column, (DataFrame, DataFrameGroupBy))
def compute_one(t, df, **kwargs):
    return df[t.columns[0]]


@dispatch(ColumnWise, DataFrame)
def compute_one(t, df, **kwargs):
    d = dict((t.child[c].scalar_symbol, df[c]) for c in t.child.columns)
    return compute(t.expr, d)


@dispatch(BinOp, Series, (Series, base))
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (Series, base), Series)
def compute_one(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, Series)
def compute_one(t, df, **kwargs):
    return getattr(np, t.symbol)(df)


@dispatch(USub, (DataFrame, Series))
def compute_one(t, df, **kwargs):
    return -df


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


@dispatch(Reduction, (DataFrame, DataFrameGroupBy, SeriesGroupBy, Series))
def compute_one(t, df, **kwargs):
    return getattr(df, t.symbol)()


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


@dispatch(By, DataFrame)
def compute_one(t, df, **kwargs):
    if t.grouper.iscolumn:
        grouper = compute(t.grouper, {t.child: df}) # a Series
    elif isinstance(t.grouper, Projection) and t.grouper.child is t.child:
        grouper = t.grouper.columns  # list of column names


    if isinstance(t.apply, Summary):
        names = t.apply.names
        preapply = DataFrame(dict(zip(
            names,
            [compute(v.child, {t.child: df}) for v in t.apply.values])))

        df2 = concat_nodup(df, preapply)

        groups = df2.groupby(grouper)

        d = defaultdict(list)
        for name, v in zip(names, t.apply.values):
            d[name].append(getattr(Series, v.symbol))

        result = groups.agg(dict(d))

        # Rearrange columns to match names order
        result = result[sorted(list(result.columns),
                               key=lambda t: names.index(t[0]))]
        result.columns = t.apply.names  # flatten down multiindex

    if isinstance(t.apply, Reduction):
        names = t.apply.dshape[0].names
        preapply = compute(t.apply.child, {t.child: df})
        # Pandas and Blaze column naming schemes differ
        # Coerce DataFrame column names to match Blaze's names
        preapply = preapply.copy()
        if isinstance(preapply, Series):
            preapply.name = names[0]
        else:
            preapply.columns = names

        df2 = concat_nodup(df, preapply)

        if t.apply.child.iscolumn:
            groups = df2.groupby(grouper)[names[0]]
        else:
            groups = df2.groupby(grouper)[names]

        result = compute_one(t.apply, groups) # do reduction

    result = DataFrame(result).reset_index()
    result.columns = t.columns
    return result


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
        return pd.concat([a, b[[c for c in b.columns
                                  if c not in a.columns]]],
                            axis=1)
    if isinstance(a, DataFrame) and isinstance(b, Series):
        if b.name not in a.columns:
            return pd.concat([a, b], axis=1)
        else:
            return a
    if isinstance(a, Series) and isinstance(b, DataFrame):
        return pd.concat([a, b[[c for c in b.columns
                                      if c != a.name]]],
                            axis=1)
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
    return s.order(t.key, ascending=t.ascending)



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
    return DataFrame(df, columns=t.columns)


@dispatch(Map, DataFrame)
def compute_one(t, df, **kwargs):
    return df.apply(lambda tup: t.func(*tup), axis=1)


@dispatch(Map, Series)
def compute_one(t, df, **kwargs):
    return df.map(t.func)


@dispatch(Apply, (Series, DataFrame))
def compute_one(t, df, **kwargs):
    return t.func(df)


@dispatch(Merge, DataFrame)
def compute_one(t, df, **kwargs):
    subexpression = common_subexpression(*t.children)
    children = [compute(child, {subexpression: df}) for child in t.children]
    return pd.concat(children, axis=1)


@dispatch(Union, DataFrame, tuple)
def compute_one(t, example, children, **kwargs):
    return pd.concat(children, axis=0)


@dispatch(Summary, DataFrame)
def compute_one(expr, data, **kwargs):
    return Series(dict(zip(expr.names,
        [compute(val, {expr.child: data}) for val in expr.values])))
