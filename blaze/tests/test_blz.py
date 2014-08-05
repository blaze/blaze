from __future__ import absolute_import, division, print_function

import blz
import numpy as np
from pandas import DataFrame

from blaze.blz import into, chunks


b = blz.btable([[1, 2, 3],
                [1., 2., 3.]],
               names=['a', 'b'])

def test_into_ndarray_btable():
    assert str(into(np.ndarray, b)) == \
            str(np.array([(1, 1.), (2, 2.), (3, 3.)],
                           dtype=[('a', int), ('b', float)]))


def test_into_btable_numpy():
    assert str(into(blz.btable, np.array([(1, 1.), (2, 2.), (3, 3.)],
                            dtype=[('a', np.int32), ('b', np.float32)]))) == \
            str(blz.btable([np.array([1, 2, 3], dtype=np.int32),
                        np.array([1., 2., 3.], dtype=np.float32)],
                       names=['a', 'b']))


def test_into_btable_DataFrame():
    df = DataFrame([[1, 'Alice'],
                    [2, 'Bob'],
                    [3, 'Charlie']], columns=['id', 'name'])

    b = into(blz.btable, df)

    assert list(b.names) == list(df.columns)
    assert list(b['id']) == [1, 2, 3]
    print(b['name'])
    print(b['name'][0])
    print(type(b['name'][0]))
    assert list(b['name']) == ['Alice', 'Bob', 'Charlie']


def test_into_btable_list():
    b = into(blz.btable, [(1, 1.), (2, 2.), (3, 3.)], names=['a', 'b'])
    assert list(b['a']) == [1, 2, 3]
    assert b.names == ['a', 'b']


def test_into_btable_iterator():
    b = into(blz.btable, iter([(1, 1.), (2, 2.), (3, 3.)]), names=['a', 'b'])
    assert list(b['a']) == [1, 2, 3]
    assert b.names == ['a', 'b']


def test_into_ndarray_barray():
    assert str(into(np.ndarray, b['a'])) == \
            str(np.array([1, 2, 3]))

def test_into_list_btable():
    assert into([], b) == [(1, 1.), (2, 2.), (3, 3.)]

def test_into_DataFrame_btable():
    result = into(DataFrame(), b)
    expected = DataFrame([[1, 1.], [2, 2.], [3, 3.]], columns=['a', 'b'])

    assert str(result) == str(expected)


def test_into_list_barray():
    assert into([], b['a']) == [1, 2, 3]
