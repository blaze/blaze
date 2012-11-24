import sys
import struct
import unittest
import os, os.path

import numpy as np

import ndtable.carray as ca
from ndtable.carray import chunk

from numpy.testing import assert_array_equal, assert_array_almost_equal

is_64bit = (struct.calcsize("P") == 8)

# Just memory tests for now

class chunkTest(unittest.TestCase):

    def test01(self):
        """Testing `__getitem()__` method with scalars"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1]->", `b[1]`
        self.assert_(a[1] == b[1], "Values in key 1 are not equal")

    def test02(self):
        """Testing `__getitem()__` method with ranges"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:3]->", `b[1:3]`
        assert_array_equal(a[1:3], b[1:3], "Arrays are not equal")

    def test03(self):
        """Testing `__getitem()__` method with ranges and steps"""
        a = np.arange(1e3)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:8:3]->", `b[1:8:3]`
        assert_array_equal(a[1:8:3], b[1:8:3], "Arrays are not equal")

    def test04(self):
        """Testing `__getitem()__` method with long ranges"""
        a = np.arange(1e4)
        b = chunk(a, atom=a.dtype, cparams=ca.cparams())
        #print "b[1:8000]->", `b[1:8000]`
        assert_array_equal(a[1:8000], b[1:8000], "Arrays are not equal")
