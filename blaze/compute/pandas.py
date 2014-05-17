"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.pandas import compute

>>> accounts = TableSymbol('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]

>>> from pandas import DataFrame
>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> df = DataFrame(data, columns=['name', 'amount'])
>>> compute(deadbeats, df)
1        Bob
2    Charlie
Name: name, dtype: object
"""
from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
import pandas
from pandas import DataFrame, Series
from multipledispatch import dispatch
import numpy as np

@dispatch(Projection, DataFrame)
def compute(t, df):
    parent = compute(t.parent, df)
    return parent[list(t.columns)]


@dispatch(Column, DataFrame)
def compute(t, df):
    parent = compute(t.parent, df)
    return parent[t.columns[0]]


@dispatch(BinOp, DataFrame)
def compute(t, df):
    lhs = compute(t.lhs, df)
    rhs = compute(t.rhs, df)
    return t.op(lhs, rhs)


@dispatch(Selection, DataFrame)
def compute(t, df):
    parent = compute(t.parent, df)
    predicate = compute(t.predicate, df)
    return parent[predicate]


@dispatch(TableSymbol, DataFrame)
def compute(t, df):
    if not list(t.columns) == list(df.columns):
        # TODO also check dtype
        raise ValueError("Schema mismatch: \n\nTable:\n%s\n\nDataFrame:\n%s"
                        % (t, df))
    return df



@dispatch(Join, DataFrame, DataFrame)
def compute(t, lhs, rhs):
    """ Join two pandas data frames on arbitrary columns

    The approach taken here could probably be improved.

    To join on two columns we force each column to be the index of the
    dataframe, perform the join, and then reset the index back to the left
    side's original index.
    """
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)
    old_left_index = lhs.index
    old_right_index = rhs.index
    if lhs.index.name:
        old_left = lhs.index.name
        rhs = lhs.reset_index()
    if rhs.index.name:
        rhs = rhs.reset_index()

    lhs = lhs.set_index(t.on_left)
    rhs = rhs.set_index(t.on_right)
    result = lhs.join(rhs)
    return result.reset_index()[t.columns]


@dispatch(UnaryOp, DataFrame)
def compute(t, s):
    parent = compute(t.parent, s)
    op = getattr(np, t.symbol)
    return op(parent)


@dispatch(Reduction, DataFrame)
def compute(t, s):
    parent = compute(t.parent, s)
    assert isinstance(parent, Series)
    op = getattr(Series, t.symbol)
    return op(parent)


@dispatch(By, DataFrame)
def compute(t, df):
    parent = compute(t.parent, df)
    grouper = compute(t.grouper, parent)
    if type(t.grouper) == Projection and t.grouper.parent == t.parent:
        grouper = list(grouper.columns)

    return parent.groupby(grouper).apply(lambda x: compute(t.apply, x))
