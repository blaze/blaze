"""

>>> from blaze.expr.table import TableSymbol
>>> from blaze.compute.python import compute

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
    return compute(t.table, df)[list(t.columns)]


@dispatch(Column, DataFrame)
def compute(t, df):
    return compute(t.table, df)[t.columns[0]]


@dispatch(BinOp, DataFrame)
def compute(t, df):
    return t.op(compute(t.lhs, df), compute(t.rhs, df))


@dispatch(Selection, DataFrame)
def compute(t, df):
    return compute(t.table, df)[compute(t.predicate, df)]


@dispatch(TableSymbol, DataFrame)
def compute(t, df):
    if not list(t.columns) == list(df.columns):
        # TODO also check dtype
        raise ValueError("Schema mismatch")
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
    else:
        old_left = None
    if rhs.index.name:
        old_right = rhs.index.name
        rhs = rhs.reset_index()
    else:
        old_right = None

    lhs = lhs.set_index(t.on_left)
    rhs = rhs.set_index(t.on_right)
    result = lhs.join(rhs)
    if old_left:
        result = result.set_index(old_left)
    return result


@dispatch(UnaryOp, DataFrame)
def compute(t, s):
    op = getattr(np, t.symbol)
    return op(compute(t.table, s))


@dispatch(Reduction, DataFrame)
def compute(t, s):
    s = compute(t.table, s)
    assert isinstance(s, Series)
    op = getattr(Series, t.symbol)
    return op(s)
