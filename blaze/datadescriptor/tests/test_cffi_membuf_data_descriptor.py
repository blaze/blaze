import unittest

import sys
from ...py3help import skipIf
import blaze
from blaze import datashape
from blaze.datadescriptor import (MemBufDataDescriptor,
                data_descriptor_from_cffi, dd_as_py,
                IDataDescriptor, IElementReader, IElementReadIter)
import ctypes
try:
    import cffi
    ffi = cffi.FFI()
except ImportError:
    cffi = None

class TestCFFIMemBufDataDescriptor(unittest.TestCase):
    @skipIf(cffi is None, 'cffi is not installed')
    def test_scalar(self):
        a = ffi.new('int *', 3)
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('int32'))
        self.assertEqual(dd_as_py(dd), 3)
        self.assertTrue(isinstance(dd_as_py(dd), int))

        a = ffi.new('float *', 3.25)
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('float32'))
        self.assertEqual(dd_as_py(dd), 3.25)
        self.assertTrue(isinstance(dd_as_py(dd), float))

    @skipIf(cffi is None, 'cffi is not installed')
    def test_1d_array(self):
        # An array where the size is in the type
        a = ffi.new('short[32]', [2*i for i in range(32)])
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, int16'))
        self.assertEqual(dd_as_py(dd), [2*i for i in range(32)])

        # An array where the size is not in the type
        a = ffi.new('double[]', [1.5*i for i in range(32)])
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, float64'))
        self.assertEqual(dd_as_py(dd), [1.5*i for i in range(32)])

    @skipIf(cffi is None, 'cffi is not installed')
    def test_2d_array(self):
        # An array where the leading array size is in the type
        vals = [[2**i + j for i in range(35)] for j in range(32)]
        a = ffi.new('long long[32][35]', vals)
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, 35, int64'))
        self.assertEqual(dd_as_py(dd), vals)

        # An array where the leading array size is not in the type
        vals = [[a + b*2 for a in range(35)] for b in range(32)]
        a = ffi.new('unsigned char[][35]', vals)
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('32, 35, uint8'))
        self.assertEqual(dd_as_py(dd), vals)

    @skipIf(cffi is None, 'cffi is not installed')
    def test_3d_array(self):
        # Simple 3D array
        vals = [[[(i + 2*j + 3*k)
                        for i in range(10)]
                        for j in range(12)]
                        for k in range(14)]
        a = ffi.new('unsigned int[14][12][10]', vals)
        dd = data_descriptor_from_cffi(ffi, a, writable=True)
        self.assertEqual(dd.dshape, blaze.dshape('14, 12, 10, uint32'))
        self.assertEqual(dd_as_py(dd), vals)

if __name__ == '__main__':
    unittest.main()

