import pytest
tables = pytest.importorskip('tables')

from blaze.utils import tmpfile
from blaze import symbol, discover, compute, pre_compute
import pandas as pd
from datetime import datetime
from odo import Chunks, odo


@pytest.fixture
def df():
    return pd.DataFrame([['a', 1, 10., datetime(2000, 1, 1)],
                         ['ab', 2, 20., datetime(2000, 2, 2)],
                         ['abc', 3, 30., datetime(2000, 3, 3)],
                         ['abcd', 4, 40., datetime(2000, 4, 4)]],
                        columns=['name', 'a', 'b', 'time'])


@pytest.fixture
def s(hdf):
    return symbol('s', discover(hdf))


@pytest.yield_fixture(params=['fixed', 'table'])
def hdf(df, request):
    with tmpfile('.hdf5') as fn:
        df.to_hdf(fn, '/data', format=request.param)
        df.to_hdf(fn, '/nested/data', format=request.param)
        with pd.HDFStore(fn, mode='r') as r:
            yield r


def test_basic_compute(hdf, s):
    result = compute(s.data, hdf)
    types = (
        pd.DataFrame,
        pd.io.pytables.Fixed,
        pd.io.pytables.AppendableFrameTable,
        Chunks
    )
    assert isinstance(result, types)


def test_pre_compute(hdf, s):
    result = pre_compute(s, hdf.get_storer('data'))
    assert isinstance(result, (pd.DataFrame, Chunks))


def test_groups(hdf, df, s):
    assert discover(hdf) == discover(dict(data=df, nested=dict(data=df)))
    assert odo(compute(s.nested.data.a, hdf), list) == [1, 2, 3, 4]
