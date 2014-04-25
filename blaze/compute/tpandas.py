"""

>>> from blaze.objects.table import Table
>>> from blaze.compute.python import compute

>>> accounts = Table('{name: string, amount: int}')
>>> deadbeats = accounts['name'][accounts['amount'] < 0]

>>> from pandas import DataFrame
>>> data = [['Alice', 100], ['Bob', -50], ['Charlie', -20]]
>>> df = DataFrame(data, columns=['name', 'amount'])
>>> compute(deadbeats, df)
1        Bob
2    Charlie
Name: name, dtype: object
"""

from blaze.objects.table import *
import pandas
from pandas import DataFrame
from multipledispatch import dispatch

base = (int, float, str, bool)

@dispatch(Projection, DataFrame)
def compute(t, df):
    return compute(t.table, df)[list(t.columns)]


@dispatch(Column, DataFrame)
def compute(t, df):
    return compute(t.table, df)[t.columns[0]]


@dispatch(base, object)
def compute(a, b):
    return a


@dispatch(Relational, DataFrame)
def compute(t, df):
    return t.op(compute(t.lhs, df), compute(t.rhs, df))


@dispatch(Selection, DataFrame)
def compute(t, df):
    return compute(t.table, df)[compute(t.predicate, df)]


@dispatch(Table, DataFrame)
def compute(t, df):
    if not list(t.columns) == list(df.columns):
        # TODO also check dtype
        raise ValueError("Schema mismatch")
    return df
