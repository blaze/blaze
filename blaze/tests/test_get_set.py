from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

import blaze
from blaze import BLZ_DDesc
from blaze.datadescriptor import ddesc_as_py


class getitem(unittest.TestCase):
    ddesc = None

    def test_scalar(self):
        a = blaze.array(np.arange(3), ddesc=self.ddesc)
        self.assertEqual(ddesc_as_py(a[0].ddesc), 0)

    def test_1d(self):
        a = blaze.array(np.arange(3), ddesc=self.ddesc)
        # print("a:", a, self.caps)
        self.assertEqual(ddesc_as_py(a[0:2].ddesc), [0,1])

    def test_2d(self):
        a = blaze.array(np.arange(3*3).reshape(3,3), ddesc=self.ddesc)
        self.assertEqual(ddesc_as_py(a[1].ddesc), [3,4,5])

class getitem_blz(getitem):
    ddesc = BLZ_DDesc(mode='w')

class setitem(unittest.TestCase):
    ddesc = None

    def test_scalar(self):
        a = blaze.array(np.arange(3), ddesc=self.ddesc)
        a[0] = 1
        self.assertEqual(ddesc_as_py(a[0].ddesc), 1)

    def test_1d(self):
        a = blaze.array(np.arange(3), ddesc=self.ddesc)
        a[0:2] = 2
        self.assertEqual(ddesc_as_py(a[0:2].ddesc), [2,2])

    def test_2d(self):
        a = blaze.array(np.arange(3*3).reshape(3,3), ddesc=self.ddesc)
        a[1] = 2
        self.assertEqual(ddesc_as_py(a[1].ddesc), [2,2,2])

class setitem_blz(getitem):
    ddesc = BLZ_DDesc(mode='w')

if __name__ == '__main__':
    unittest.main()
