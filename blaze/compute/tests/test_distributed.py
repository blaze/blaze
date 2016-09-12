from __future__ import absolute_import, division, print_function

import pytest
import sys
from pandas import DataFrame, Series

dist = pytest.importorskip('distributed')

import numpy as np
from distributed.client import Client
from distributed.deploy import LocalCluster
from blaze import compute, symbol
from blaze.compute.distributed import DataFrame as DDataFrame
from blaze.compute.distributed import Series as DSeries


# Work around https://github.com/dask/distributed/issues/516
cluster = LocalCluster(nanny=False)
client = Client(cluster)
#client = Client()

df = DataFrame([['Alice', 100, 1],
                ['Bob', 200, 2],
                ['Alice', 50, 3]], columns=['name', 'amount', 'id'])
dft = symbol('t', 'var * {name: string, amount: int, id: int}')

s = Series([1, 2, 3], name='a')
st = symbol('t', 'var * {a: int64}')

def test_simple():

    def func(data): return data.amount * data.id
    result = client.submit(func, df)
    expected = df.amount * df.id
    assert (result.result() == expected).all()


# TODO: The test_series tests now fail because the compute_up(Field, Future)
#       function expects `Future` to hold a DataFrame,
#       for which a field is computed differently from Series...
#       Unfortunately there is no way to query a future's type until it has completed its computation...
@pytest.mark.parametrize('expr',
                         [st.a * 2,
                          st.a + st.a])
def test_series(expr):

    ds, = client.scatter([s])
    ds = DSeries(ds)
    expected = compute(expr, s)
    result = compute(expr, ds)
    assert (result.result() == expected).all()


@pytest.mark.parametrize('expr',
                         [dft.amount * dft.id,
                          dft.amount + 1])
def test_dataframe(expr):
    
    ddf, = client.scatter([df])
    ddf = DDataFrame(ddf)
    expected = compute(expr, df)
    result = compute(expr, ddf)
    assert (result.result() == expected).all()
