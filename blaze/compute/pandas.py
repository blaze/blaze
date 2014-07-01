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

from ..dispatch import dispatch
from ..expr.table import *
from ..expr.scalar import UnaryOp, BinOp
from .core import compute, compute_one, base
from . import core

__all__ = ['compute_one']


@dispatch(Projection, DataFrame)
def compute_one(t, df):
    return df[list(t.columns)]


@dispatch(Column, (DataFrame, DataFrameGroupBy))
def compute_one(t, df):
    return df[t.columns[0]]


@dispatch(ColumnWise, DataFrame)
def compute_one(t, df):
    columns = [t.parent[c] for c in t.parent.columns]
    d = dict((t.parent[c].scalar_symbol, df[c]) for c in t.parent.columns)
    return compute(t.expr, d)


@dispatch(BinOp, Series, (Series, base))
def compute_one(t, lhs, rhs):
    return t.op(lhs, rhs)


@dispatch(BinOp, (Series, base), Series)
def compute_one(t, lhs, rhs):
    return t.op(lhs, rhs)


@dispatch(UnaryOp, Series)
def compute_one(t, df):
    return getattr(np, t.symbol)(df)


@dispatch(Neg, (DataFrame, Series))
def compute_one(t, df):
    return -df


@dispatch(Selection, DataFrame)
def compute_one(t, df):
    predicate = compute(t.predicate, {t.parent: df})
    return df[predicate]


@dispatch(TableSymbol, DataFrame)
def compute_one(t, df):
    if not list(t.columns) == list(df.columns):
        # TODO also check dtype
        raise ValueError("Schema mismatch: \n\nTable:\n%s\n\nDataFrame:\n%s"
                        % (t, df))
    return df


@dispatch(Join, DataFrame, DataFrame)
def compute_one(t, lhs, rhs):
    """ Join two pandas data frames on arbitrary columns

    The approach taken here could probably be improved.

    To join on two columns we force each column to be the index of the
    dataframe, perform the join, and then reset the index back to the left
    side's original index.
    """
    old_left_index = lhs.index
    old_right_index = rhs.index
    if lhs.index.name:
        old_left = lhs.index.name
        rhs = lhs.reset_index()
    if rhs.index.name:
        rhs = rhs.reset_index()

    lhs = lhs.set_index(t.on_left)
    rhs = rhs.set_index(t.on_right)
    result = lhs.join(rhs, how='inner')
    return result.reset_index()[t.columns]


@dispatch(TableSymbol, (DataFrameGroupBy, SeriesGroupBy))
def compute_one(t, gb):
    return gb


@dispatch(Reduction, (DataFrame, DataFrameGroupBy, SeriesGroupBy, Series))
def compute_one(t, df):
    return getattr(df, t.symbol)()


@dispatch(Distinct, DataFrame)
def compute_one(t, df):
    return df.drop_duplicates()


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
def compute_one(t, df):
    assert isinstance(t.apply, Reduction)
    grouper = DataFrame(compute(t.grouper, {t.parent: df}))
    pregrouped = DataFrame(compute(t.apply.parent, {t.parent: df}))

    full = grouper.join(pregrouped)
    groups = full.groupby(unpack(grouper.columns))[unpack(pregrouped.columns)]

    g = TableSymbol('group', t.apply.parent.schema)
    reduction = t.apply.subs({t.apply.parent: g})
    result = compute(reduction, {g: groups})

    if isinstance(result, Series):
        result.name = unpack(pregrouped.columns)
        result = DataFrame(result)

    return result[list(pregrouped.columns)].reset_index()


@dispatch(Sort, DataFrame)
def compute_one(t, df):
    return df.sort(t.column, ascending=t.ascending)


@dispatch(Sort, Series)
def compute_one(t, s):
    s = s.copy()
    s.sort(t.column, ascending=t.ascending)
    return s


@dispatch(Head, DataFrame)
def compute_one(t, df):
    return df.head(t.n)


@dispatch(Label, DataFrame)
def compute_one(t, df):
    return DataFrame(df, columns=[t.label])


@dispatch(Label, Series)
def compute_one(t, df):
    return Series(df, name=t.label)


@dispatch(ReLabel, DataFrame)
def compute_one(t, df):
    return DataFrame(df, columns=t.columns)


@dispatch(Map, DataFrame)
def compute_one(t, df):
    return df.apply(lambda tup: t.func(*tup), axis=1)


@dispatch(Map, Series)
def compute_one(t, df):
    return df.map(t.func)


@dispatch(Apply, (Series, DataFrame))
def compute_one(t, df):
    return t.func(df)


@dispatch(Merge, DataFrame)
def compute_one(t, df):
    ancestor = common_ancestor(*t.children)
    children = [compute(child, {ancestor: df}) for child in t.children]
    return pd.concat(children, axis=1)
