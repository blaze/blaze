from blaze.compute.hdfstore import *
from blaze.utils import tmpfile
from blaze import symbol, discover, compute
import pandas as pd
from datetime import datetime
from into import Chunks, resource, into
from blaze.compatibility import skipif, PY3


df = pd.DataFrame([['a', 1, 10., datetime(2000, 1, 1)],
                   ['ab', 2, 20., datetime(2000, 2, 2)],
                   ['abc', 3, 30., datetime(2000, 3, 3)],
                   ['abcd', 4, 40., datetime(2000, 4, 4)]],
                   columns=['name', 'a', 'b', 'time'])


@skipif(PY3, reason="https://github.com/pydata/pandas/issues/9219")
def test_hdfstore():
    with tmpfile('.hdf5') as fn:
        df.to_hdf(fn, '/appendable', format='table')
        df.to_hdf(fn, '/fixed')

        hdf = resource('hdfstore://%s' % fn)
        s = symbol('s', discover(hdf))

        assert isinstance(compute(s.fixed, hdf),
                          (pd.DataFrame, pd.io.pytables.Fixed))
        assert isinstance(compute(s.appendable, hdf),
                          (pd.io.pytables.AppendableFrameTable, Chunks))

        s = symbol('s', discover(df))
        f = resource('hdfstore://%s::/fixed' % fn)
        a = resource('hdfstore://%s::/appendable' % fn)
        assert isinstance(pre_compute(s, a), Chunks)

        hdf.close()
        f.parent.close()
        a.parent.close()



def test_groups():
    with tmpfile('.hdf5') as fn:
        df.to_hdf(fn, '/data/fixed')

        hdf = resource('hdfstore://%s' % fn)
        assert discover(hdf) == discover({'data': {'fixed': df}})

        s = symbol('s', discover(hdf))

        assert list(compute(s.data.fixed, hdf).a) == [1, 2, 3, 4]

        hdf.close()
