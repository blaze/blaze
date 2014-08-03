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

__all__ = ['compute_one']


@dispatch(Projection, DataFrame)
def compute_one(t, df, **kwargs):
    return df[list(t.columns)]


@dispatch(Column, (DataFrame, DataFrameGroupBy))
def compute_one(t, df, **kwargs):
    return df[t.columns[0]]


@dispatch(ColumnWise, DataFrame)
def compute_one(t, df, **kwargs):
    columns = [t.child[c] for c in t.child.columns]
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
    assert isinstance(t.apply, Reduction)
    grouper = DataFrame(compute(t.grouper, {t.child: df}))
    pregrouped = DataFrame(compute(t.apply.child, {t.child: df}))

    full = grouper.join(pregrouped)
    groups = full.groupby(unpack(grouper.columns))[unpack(pregrouped.columns)]

    g = TableSymbol('group', t.apply.child.schema)
    reduction = t.apply.subs({t.apply.child: g})
    result = compute(reduction, {g: groups})

    if isinstance(result, Series):
        result.name = unpack(pregrouped.columns)
        result = DataFrame(result)

    return result[list(pregrouped.columns)].reset_index()


@dispatch(Sort, DataFrame)
def compute_one(t, df, **kwargs):
    return df.sort(t.key, ascending=t.ascending)


@dispatch(Sort, Series)
def compute_one(t, s, **kwargs):
    return s.order(t.key, ascending=t.ascending)


@dispatch(Head, DataFrame)
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
