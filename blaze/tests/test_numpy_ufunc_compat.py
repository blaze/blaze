from __future__ import absolute_import, division, print_function

import unittest
import blaze
from blaze.datadescriptor import dd_as_py
import numpy as np

class TestBitwiseOps(unittest.TestCase):
    def test_bitwise_or_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t | t)._data), True)
        self.assertEqual(dd_as_py((t | f)._data), True)
        self.assertEqual(dd_as_py((f | t)._data), True)
        self.assertEqual(dd_as_py((f | f)._data), False)

    def test_bitwise_or_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a | b)._data), x | y)
        self.assertEqual(dd_as_py(blaze.bitwise_or(a, b)._data), x | y)

    def test_bitwise_and_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t & t)._data), True)
        self.assertEqual(dd_as_py((t & f)._data), False)
        self.assertEqual(dd_as_py((f & t)._data), False)
        self.assertEqual(dd_as_py((f & f)._data), False)

    def test_bitwise_and_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a & b)._data), x & y)
        self.assertEqual(dd_as_py(blaze.bitwise_and(a, b)._data), x & y)

    def test_bitwise_xor_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((t ^ t)._data), False)
        self.assertEqual(dd_as_py((t ^ f)._data), True)
        self.assertEqual(dd_as_py((f ^ t)._data), True)
        self.assertEqual(dd_as_py((f ^ f)._data), False)

    def test_bitwise_xor_uint64(self):
        x, y = 0x3192573469a2b3a1, 0x9274a2e219c27638
        a = blaze.array(x, 'uint64')
        b = blaze.array(y, 'uint64')
        self.assertEqual(dd_as_py((a ^ b)._data), x ^ y)
        self.assertEqual(dd_as_py(blaze.bitwise_xor(a, b)._data), x ^ y)

    def test_bitwise_not_bool(self):
        t = blaze.array(True)
        f = blaze.array(False)
        self.assertEqual(dd_as_py((~t)._data), False)
        self.assertEqual(dd_as_py((~f)._data), True)

    def test_bitwise_not_uint64(self):
        x = 0x3192573469a2b3a1
        a = blaze.array(x, 'uint64')
        self.assertEqual(dd_as_py((~a)._data), x ^ 0xffffffffffffffff)
        self.assertEqual(dd_as_py(blaze.bitwise_not(a)._data),
                         x ^ 0xffffffffffffffff)

