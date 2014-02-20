from __future__ import absolute_import, division, print_function

import unittest
import ctypes

import blaze
from blaze.datadescriptor import data_descriptor_from_ctypes


class TestArrayStr(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(str(blaze.array(100)), '100')
        self.assertEqual(str(blaze.array(-3.25)), '-3.25')
        self.assertEqual(str(blaze.array(True)), 'True')
        self.assertEqual(str(blaze.array(False)), 'False')

    def test_deferred_scalar(self):
        a = blaze.array(3) + blaze.array(5)
        self.assertEqual(str(a), '8')

    def test_ctypes_scalar(self):
        dd = data_descriptor_from_ctypes(ctypes.c_int32(1022), writable=True)
        a = blaze.array(dd)
        self.assertEqual(str(a), '1022')

    def test_1d_array(self):
        self.assertEqual(str(blaze.array([1,2,3])), '[1 2 3]')

    def test_ctypes_1d_array(self):
        cdat = (ctypes.c_int64 * 3)()
        cdat[0] = 3
        cdat[1] = 6
        cdat[2] = 10
        dd = data_descriptor_from_ctypes(cdat, writable=True)
        a = blaze.array(dd)
        self.assertEqual(str(a), '[ 3  6 10]')

    def test_ragged_array(self):
        a = blaze.array([[1,2,3],[4,5]])
        self.assertEqual(str(a),
            '[[        1         2         3]\n [        4         5]]')

    def test_empty_array(self):
        a = blaze.array([[], []])
        self.assertEqual(str(a), '[[]\n []]')
        a = blaze.array([[], [1, 2]])
        self.assertEqual(str(a), '[[]\n [     1      2]]')

    def test_str_array(self):
        # Basically check that it doesn't raise an exception to
        # get the string
        a = blaze.array(['this', 'is', 'a', 'test'])
        self.assertTrue(str(a) != '')
        self.assertTrue(repr(a) != '')

    def test_struct_array(self):
        # Basically check that it doesn't raise an exception to
        # get the string
        a = blaze.array([(1, 2), (3, 4), (5, 6)],
                dshape='{x: int32; y: float64}')
        self.assertTrue(str(a) != '')
        self.assertTrue(repr(a) != '')

if __name__ == '__main__':
    unittest.main(verbosity=2)
