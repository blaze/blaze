from __future__ import absolute_import, division, print_function

import bcolz
import numpy as np
from pandas import DataFrame
from toolz import count

from blaze.bcolz import into, chunks


b = bcolz.ctable([[1, 2, 3],
                [1., 2., 3.]],
               names=['a', 'b'])

def test_into_ndarray_ctable():
    assert str(into(np.ndarray, b)) == \
            str(np.array([(1, 1.), (2, 2.), (3, 3.)],
                           dtype=[('a', int), ('b', float)]))


def test_into_ctable_numpy():
    assert str(into(bcolz.ctable, np.array([(1, 1.), (2, 2.), (3, 3.)],
                            dtype=[('a', np.int32), ('b', np.float32)]))) == \
            str(bcolz.ctable([np.array([1, 2, 3], dtype=np.int32),
                        np.array([1., 2., 3.], dtype=np.float32)],
                       names=['a', 'b']))


def test_into_ctable_DataFrame():
    df = DataFrame([[1, 'Alice'],
                    [2, 'Bob'],
                    [3, 'Charlie']], columns=['id', 'name'])

    b = into(bcolz.ctable, df)

    assert list(b.names) == list(df.columns)
    assert list(b['id']) == [1, 2, 3]
    print(b['name'])
    print(b['name'][0])
    print(type(b['name'][0]))
    assert list(b['name']) == ['Alice', 'Bob', 'Charlie']


def test_into_ctable_list():
    b = into(bcolz.ctable, [(1, 1.), (2, 2.), (3, 3.)], names=['a', 'b'])
    assert list(b['a']) == [1, 2, 3]
    assert b.names == ['a', 'b']


def test_into_ctable_list_datetimes():
    from datetime import datetime
    b = into(bcolz.carray, [datetime(2012, 1, 1), datetime(2013, 2, 2)])
    assert np.issubdtype(b.dtype, np.datetime64)


def test_into_ctable_iterator():
    b = into(bcolz.ctable, iter([(1, 1.), (2, 2.), (3, 3.)]), names=['a', 'b'])
    assert list(b['a']) == [1, 2, 3]
    assert b.names == ['a', 'b']


def test_into_ndarray_carray():
    assert str(into(np.ndarray, b['a'])) == \
            str(np.array([1, 2, 3]))

def test_into_list_ctable():
    assert into([], b) == [(1, 1.), (2, 2.), (3, 3.)]

def test_into_DataFrame_ctable():
    result = into(DataFrame(), b)
    expected = DataFrame([[1, 1.], [2, 2.], [3, 3.]], columns=['a', 'b'])

    assert str(result) == str(expected)


def test_into_list_carray():
    assert into([], b['a']) == [1, 2, 3]


def test_chunks():
    x = np.array([(int(i), float(i)) for i in range(100)],
                 dtype=[('a', np.int32), ('b', np.float32)])
    b = bcolz.ctable(x)

    assert count(chunks(b, chunksize=10)) == 10
    assert (next(chunks(b, chunksize=10)) == x[:10]).all()


def test_into_chunks():
    from blaze.compute.numpy import chunks, compute_one
    from blaze.compute.chunks import chunks, compute_one, ChunkIter
    from blaze import into
    x = np.array([(int(i), float(i)) for i in range(100)],
                 dtype=[('a', np.int32), ('b', np.float32)])
    cs = chunks(x, chunksize=10)

    b1 = into(bcolz.ctable, ChunkIter(cs))
    b2 = into(bcolz.ctable, x)

    assert str(b1) == str(b2)
