import sys
import unittest
import ctypes
from ..py2help import skip
import blaze
from blaze.datadescriptor import data_descriptor_from_ctypes

from dynd import nd, ndt
from blaze.datadescriptor import DyNDDataDescriptor

class TestArrayStr(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(str(blaze.array(100)), '100')
        self.assertEqual(str(blaze.array(-3.25)), '-3.25')
        self.assertEqual(str(blaze.array(True)), 'True')
        self.assertEqual(str(blaze.array(False)), 'False')

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

    # This is broken, needs to be revisited
    @skip('ragged array printing not working')
    def test_ragged_array(self):
        dd = DyNDDataDescriptor(nd.array([[1,2,3],[4,5]]))
        a = blaze.array(dd)
        self.assertEqual(str(a),
            '[[        1         2         3]\n [        4         5]]')

if __name__ == '__main__':
    unittest.main(verbosity=2)
