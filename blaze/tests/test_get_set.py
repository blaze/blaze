from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import blaze
from blaze.datadescriptor import dd_as_py
from blaze.py2help import skip


class getitem(unittest.TestCase):
    caps={'compress': False}  # the default is non-compressed arrays

    def test_scalar(self):
        a = blaze.array(np.arange(3), caps=self.caps)
        self.assertEqual(dd_as_py(a[0]._data), 0)

    @skip('slices should implemented')
    def test_1d(self):
        a = blaze.array(np.arange(3), caps=self.caps)
        print("a:", a, self.caps)
        self.assertEqual(dd_as_py(a[0:2]._data), [0,1])

    def test_2d(self):
        a = blaze.array(np.arange(3*3).reshape(3,3), caps=self.caps)
        self.assertEqual(dd_as_py(a[1]._data), [3,4,5])

class getitem_blz(getitem):
    caps={'compress': True}

class setitem(unittest.TestCase):
    caps={'compress': False}  # the default is non-compressed arrays

    def test_scalar(self):
        a = blaze.array(np.arange(3), caps=self.caps)
        a[0] = 1
        self.assertEqual(dd_as_py(a[0]._data), 1)

    @skip('slices should be implemented')
    def test_1d(self):
        a = blaze.array(np.arange(3), caps=self.caps)
        a[0:2] = 2
        self.assertEqual(dd_as_py(a[0:2]._data), [2,2])

    def test_2d(self):
        a = blaze.array(np.arange(3*3).reshape(3,3), caps=self.caps)
        a[1] = 2
        self.assertEqual(dd_as_py(a[1]._data), [2,2,2])

# BLZ is going to be read and append only for the time being
# class setitem_blz(setitem):
#     caps={'compress': True}

if __name__ == '__main__':
    unittest.main()
