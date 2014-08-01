from __future__ import absolute_import, division, print_function

import blz
import numpy as np

from blaze.blz import into


b = blz.btable([[1, 2, 3],
                [1., 2., 3.]],
               names=['a', 'b'])

def test_into_ndarray():
    assert str(into(np.ndarray, b)) == \
            str(np.array([(1, 1.), (2, 2.), (3, 3.)],
                           dtype=[('a', int), ('b', float)]))

def test_into_list():
    assert into([], b) == [(1, 1.), (2, 2.), (3, 3.)]
