import unittest

import sys
import blaze
from blaze import datashape
from blaze.datadescriptor import (MemBufDataDescriptor,
                data_descriptor_from_ctypes, dd_as_py,
                IDataDescriptor, IElementReader, IElementReadIter)
import ctypes

class TestCTypesMemBufDataDescriptor(unittest.TestCase):
    def test_scalar(self):
        a = ctypes.c_int(3)
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('int32'))
        self.assertEqual(dd_as_py(dd), 3)
        self.assertTrue(isinstance(dd_as_py(dd), int))

        a = ctypes.c_float(3.25)
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('float32'))
        self.assertEqual(dd_as_py(dd), 3.25)
        self.assertTrue(isinstance(dd_as_py(dd), float))

    def test_1d_array(self):
        a = (ctypes.c_short * 32)()
        for i in range(32):
            a[i] = 2*i
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, int16'))
        self.assertEqual(dd_as_py(dd), [2*i for i in range(32)])

        a = (ctypes.c_double * 32)()
        for i in range(32):
            a[i] = 1.5*i
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, float64'))
        self.assertEqual(dd_as_py(dd), [1.5*i for i in range(32)])

    def test_2d_array(self):
        a = (ctypes.c_double * 35 * 32)()
        vals = [[2**i + j for i in range(35)] for j in range(32)]
        for i in range(32):
            for j in range(35):
                a[i][j] = vals[i][j]
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, 35, float64'))
        self.assertEqual(dd_as_py(dd), vals)

        a = (ctypes.c_uint8 * 35 * 32)()
        vals = [[i + j*2 for i in range(35)] for j in range(32)]
        for i in range(32):
            for j in range(35):
                a[i][j] = vals[i][j]
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, 35, uint8'))
        self.assertEqual(dd_as_py(dd), vals)

    def test_3d_array(self):
        # Simple 3D array
        a = (ctypes.c_uint32 * 10 * 12 * 14)()
        vals = [[[(i + 2*j + 3*k)
                        for i in range(10)]
                        for j in range(12)]
                        for k in range(14)]
        for i in range(14):
            for j in range(12):
                for k in range(10):
                    a[i][j][k] = vals[i][j][k]
        dd = data_descriptor_from_ctypes(a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('14, 12, 10, uint32'))
        self.assertEqual(dd_as_py(dd), vals)

if __name__ == '__main__':
    unittest.main()

