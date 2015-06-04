import os
import json

from datetime import datetime

import numpy as np
import pandas as pd
import h5py

from datashape import dshape

from blaze import discover
from blaze.utils import tmpfile, json_dumps, spider


def test_tmpfile():
    with tmpfile() as f:
        with open(f, 'w') as a:
            a.write('')
        with tmpfile() as g:
            assert f != g

    assert not os.path.exists(f)


def test_json_encoder():
    result = json.dumps([1, datetime(2000, 1, 1, 12, 30, 0)],
                        default=json_dumps)
    assert result == '[1, "2000-01-01T12:30:00Z"]'
    assert json.loads(result) == [1, "2000-01-01T12:30:00Z"]


def test_spider(tmpdir):
    csvf = tmpdir.join('foo.csv')
    csvf.write('a,b\n1,2\n3,4')
    h5f = tmpdir.join('foo.hdf5')
    data = np.random.randn(10, 2)
    with h5py.File(str(h5f)) as f:
        f.create_dataset(name='fooh5', shape=data.shape,
                         dtype=data.dtype, data=data)
    jsonf = tmpdir.mkdir('sub').join('foo.json')
    jsonf.write(json.dumps({'a': 2,
                            'b': 3,
                            'c': str(pd.Timestamp('now'))}))
    result = spider(str(tmpdir))
    ss = """{
    %r: {
        'foo.csv': var * {a: int64, b: int64},
        'foo.hdf5': {fooh5: 10 * 2 * float64},
        sub: {'foo.json': {a: int64, b: int64, c: datetime}}
    }
}""" % os.path.basename(str(tmpdir))
    assert dshape(discover(result)) == dshape(ss)
