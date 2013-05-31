import unittest
import sys
import blaze
from blaze import datashape
from blaze.datadescriptor import (DyNDDataDescriptor, data_descriptor_from_ctypes,
                IDataDescriptor, IElementReader, IElementReadIter,
                IElementWriter, IElementWriteIter,
                dd_as_py, execute_unary_single)
from blaze.ckernel import CKernel, UnarySingleOperation
from blaze.py3help import _inttypes, skipIf, izip
import ctypes

try:
    import dynd
    from dynd import nd, ndt, lowlevel
except ImportError:
    dynd = None

class TestDyNDBroadcastCKernel(unittest.TestCase):
    def test_assign_scalar(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_int64(1028), writable=False)
        dst = data_descriptor_from_ctypes(ctypes.c_float(-100), writable=True)
        # Get a kernel from dynd
        ck = CKernel(UnarySingleOperation)
        lowlevel.py_api.make_assignment_kernel(
                        ndt.float32, ndt.int64, 'single',
                        ctypes.addressof(ck.dynamic_kernel_instance))
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, ck)
        self.assertEqual(dd_as_py(dst), 1028.0)

    def test_assign_scalar_to_array(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_int64(1028), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3)(), writable=True)
        # Get a kernel from dynd
        ck = CKernel(UnarySingleOperation)
        lowlevel.py_api.make_assignment_kernel(
                        ndt.float32, ndt.int64, 'single',
                        ctypes.addressof(ck.dynamic_kernel_instance))
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, ck)
        self.assertEqual(dd_as_py(dst), [1028.0, 1028.0, 1028.0])

if __name__ == '__main__':
    unittest.main()

