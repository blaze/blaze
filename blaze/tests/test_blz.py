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
