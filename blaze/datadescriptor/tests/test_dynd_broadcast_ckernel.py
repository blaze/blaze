import unittest
import sys
import ctypes

import blaze
from blaze import datashape
from blaze.datadescriptor import (DyNDDataDescriptor, data_descriptor_from_ctypes,
                IDataDescriptor, IElementReader, IElementReadIter,
                IElementWriter, IElementWriteIter,
                dd_as_py, execute_unary_single)
from blaze.ckernel import CKernel, UnarySingleOperation
from blaze.py3help import _inttypes, skipIf, izip

try:
    import dynd
    from dynd import nd, ndt, _lowlevel
except ImportError:
    dynd = None

class TestDyNDBroadcastCKernel(unittest.TestCase):
    def setUp(self):
        if dynd is not None:
            # Get a kernel from dynd
            self.ck = CKernel(UnarySingleOperation)
            _lowlevel.py_api.make_assignment_kernel(
                            ndt.float32, ndt.int64, 'single',
                            ctypes.addressof(self.ck.dynamic_kernel_instance))

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_scalar(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_int64(1028), writable=False)
        dst = data_descriptor_from_ctypes(ctypes.c_float(-100), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), 1028.0)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_scalar_to_one_d_array(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_int64(1028), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [1028.0] * 3)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_scalar_to_two_d_array(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_int64(-50), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [[-50.0] * 3] * 4)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_one_d_to_one_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_int64 * 3)()
        for i, val in enumerate([3, 6, 9]):
            src_data[i] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [3.0, 6.0, 9.0])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_one_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_int64 * 3)()
        for i, val in enumerate([3, 6, 9]):
            src_data[i] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [[3.0, 6.0, 9.0]] * 4)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_int64 * 3 * 4)()
        for i, val_i in enumerate([[3, 6, 9], [1, 2, 3], [7,3,4], [12, 10, 2]]):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [[3, 6, 9], [1, 2, 3], [7,3,4], [12, 10, 2]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_broadcast_inner_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_int64 * 1 * 4)()
        for i, val_i in enumerate([[3], [1], [7], [12]]):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [[i]*3 for i in [3,1,7,12]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_assign_broadcast_outer_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_int64 * 3 * 1)()
        for i, val_i in enumerate([[3,1,7]]):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_float * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float32, datashape.int64, self.ck)
        self.assertEqual(dd_as_py(dst), [[3,1,7]]*4)

if __name__ == '__main__':
    unittest.main()

