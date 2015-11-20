import numpy as np
import pandas as pd
import pandas.util.testing as tm

import dask.dataframe as dd

from blaze.expr import symbol
from blaze.compute.core import compute
from blaze.compatibility import builtins


def eq(a, b):
    if isinstance(a, dd.DataFrame):
        tm.assert_frame_equal(a.compute(), b)
    elif isinstance(b, dd.Series):
        tm.assert_series_equal(a.compute(), b)

t = symbol('t', 'var * {name: string, amount: int, id: int}')
nt = symbol('t', 'var * {name: ?string, amount: float64, id: int}')

df = pd.DataFrame([['Alice', 100, 1],
                   ['Bob', 200, 2],
                   ['Alice', 50, 3]], columns=['name', 'amount', 'id'])

ndf = pd.DataFrame([['Alice', 100.0, 1],
                    ['Bob', np.nan, 2],
                    [np.nan, 50.0, 3]], columns=['name', 'amount', 'id'])

tbig = symbol('tbig',
              'var * {name: string, sex: string[1], amount: int, id: int}')

dfbig = pd.DataFrame([['Alice', 'F', 100, 1],
                      ['Alice', 'F', 100, 3],
                      ['Drew', 'F', 100, 4],
                      ['Drew', 'M', 100, 5],
                      ['Drew', 'M', 200, 5]],
                     columns=['name', 'sex', 'amount', 'id'])

ddf = dd.from_pandas(df, npartitions=2)
ddfbig = dd.from_pandas(dfbig, npartitions=2)


def test_symbol():
    eq(compute(t, ddf), df)

def test_head():
    eq(compute(t.head(1), ddf), df.head(1))


def test_tail():
    eq(compute(t.tail(1), ddf), df.tail(1))


def test_relabel():
    result = compute(t.relabel({'name': 'NAME', 'id': 'ID'}), ddf)
    expected = df.rename(columns={'name': 'NAME', 'id': 'ID'})
    eq(result, expected)


def test_relabel_series():
    result = compute(t.relabel({'name': 'NAME'}), ddf.name)
    assert result.name == 'NAME'

def test_series_columnwise():
    data = pd.Series([1, 2, 3, 4], name='a')
    s = dd.from_pandas(data, npartitions=2)
    t = symbol('t', 'var * {a: int64}')
    result = compute(t.a + 1, s)
    eq(result, data)
