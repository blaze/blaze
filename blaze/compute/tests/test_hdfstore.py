import pytest
tables = pytest.importorskip('tables')

from blaze.compute.hdfstore import *
from blaze.utils import tmpfile
from datashape import discover, dshape
from blaze import symbol, compute, data
import pandas as pd
from datetime import datetime
from odo import Chunks, into
import os


try:
    f = pd.HDFStore('foo')
except (RuntimeError, ImportError) as e:
    pytest.skip('skipping test_hdfstore.py %s' % e)
else:
    f.close()
    os.remove('foo')


df = pd.DataFrame([['a', 1, 10., datetime(2000, 1, 1)],
                   ['ab', 2, 20., datetime(2000, 2, 2)],
                   ['abc', 3, 30., datetime(2000, 3, 3)],
                   ['abcd', 4, 40., datetime(2000, 4, 4)]],
                   columns=['name', 'a', 'b', 'time'])


def test_hdfstore():
    with tmpfile('.hdf5') as fn:
        df.to_hdf(fn, '/appendable', format='table')
        df.to_hdf(fn, '/fixed')

        hdf = data('hdfstore://%s' % fn)
        s = symbol('s', discover(hdf))

        assert isinstance(compute(s.fixed, hdf),
                          (pd.DataFrame, pd.io.pytables.Fixed))
        assert isinstance(compute(s.appendable, hdf),
                          (pd.io.pytables.AppendableFrameTable, Chunks))

        s = symbol('s', discover(df))
        f = data('hdfstore://%s::/fixed' % fn)
        a = data('hdfstore://%s::/appendable' % fn)
        assert isinstance(pre_compute(s, a), Chunks)

        hdf.data.close()
        f.data.parent.close()
        a.data.parent.close()



def test_groups():
    with tmpfile('.hdf5') as fn:
        df.to_hdf(fn, '/data/fixed')

        hdf = data('hdfstore://%s' % fn)
        assert dshape(discover(hdf)) == dshape(discover({'data': {'fixed': df}}))

        s = symbol('s', discover(hdf))

        assert list(compute(s.data.fixed, hdf).a) == [1, 2, 3, 4]

        hdf.data.close()
