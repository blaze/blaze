import sys
import os
import json

import pytest

import h5py
import numpy as np
import pandas as pd

from datashape import dshape

from blaze import data_spider, discover


@pytest.fixture
def data(tmpdir):
    csvf = tmpdir.join('foo.csv')
    csvf.write('a,b\n1,2\n3,4')
    h5f = tmpdir.join('foo.hdf5')
    data = np.random.randn(10, 2)
    with h5py.File(str(h5f)) as f:
        f.create_dataset(name='fooh5', shape=data.shape,
                         dtype=data.dtype, data=data)
    jsonf = tmpdir.mkdir('sub').join('foo.json')
    jsonf.write(json.dumps([{'a': 2,
                             'b': 3.14,
                             'c': str(pd.Timestamp('now'))},
                            {'a': 2,
                             'b': 4.2,
                             'c': None,
                             'd': 'foobar'}]))
    return tmpdir


@pytest.fixture
def data_with_cycle(data):
    data.join('cycle').mksymlinkto(data)
    return data


def test_data_spider(data):
    result = data_spider(str(data))
    ss = """{
    %r: {
        'foo.csv': var * {a: int64, b: int64},
        'foo.hdf5': {fooh5: 10 * 2 * float64},
        sub: {'foo.json': 2 * {a: int64, b: float64, c: ?datetime, d: ?string}}
    }
}""" % os.path.basename(str(data))
    assert dshape(discover(result)) == dshape(ss)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='Windows does not have symlinks')
def test_data_spider_cycle(data_with_cycle):
    result = data_spider(str(data_with_cycle), followlinks=True)
    ss = """{
    %r: {
        'foo.csv': var * {a: int64, b: int64},
        'foo.hdf5': {fooh5: 10 * 2 * float64},
        sub: {'foo.json': 2 * {a: int64, b: float64, c: ?datetime, d: ?string}}
    }
}""" % os.path.basename(str(data_with_cycle))
    assert dshape(discover(result)) != dshape(ss)
