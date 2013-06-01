import unittest
import sys
import ctypes

import blaze
from blaze import datashape
from blaze import ckernel
from blaze.datadescriptor import (data_descriptor_from_ctypes,
                execute_unary_single, dd_as_py)

class TestBroadcastUnarySingleCKernel(unittest.TestCase):
    def setUp(self):
        # Create a unary single ckernel for the tests to use
        def my_kernel_func(dst_ptr, src_ptr, kdp):
            dst = ctypes.c_double.from_address(dst_ptr)
            src = ctypes.c_float.from_address(src_ptr)
            dst.value = src.value * src.value
        my_callback = ckernel.UnarySingleOperation(my_kernel_func)
        # The ctypes callback object is both the function and the owner
        self.sqr = ckernel.wrap_ckernel_func(my_callback, my_callback)

    def test_assign_scalar(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_float(1028), writable=False)
        dst = data_descriptor_from_ctypes(ctypes.c_double(-100), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), 1028.0*1028.0)

    def test_assign_scalar_to_one_d_array(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_float(1028), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [1028.0*1028.0] * 3)

    def test_assign_scalar_to_two_d_array(self):
        # Set up our data buffers
        src = data_descriptor_from_ctypes(ctypes.c_float(-50), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [[-50.0 * -50] * 3] * 4)

    def test_assign_one_d_to_one_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_float * 3)()
        for i, val in enumerate([3, 6, 9]):
            src_data[i] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [x*x for x in [3.0, 6.0, 9.0]])

    def test_assign_one_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_float * 3)()
        for i, val in enumerate([3, 6, 9]):
            src_data[i] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [[x*x for x in [3.0, 6.0, 9.0]]] * 4)

    def test_assign_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_float * 3 * 4)()
        src_list = [[3, 6, 9], [1, 2, 3], [7,3,4], [12, 10, 2]]
        for i, val_i in enumerate(src_list):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [[x*x for x in y] for y in src_list])

    def test_assign_broadcast_inner_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_float * 1 * 4)()
        src_list = [[3], [1], [7], [12]]
        for i, val_i in enumerate(src_list):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [[x*x for x in y] * 3 for y in src_list])

    def test_assign_broadcast_outer_two_d_to_two_d_array(self):
        # Set up our data buffers
        src_data = (ctypes.c_float * 3 * 1)()
        src_list =[[3,1,7]]
        for i, val_i in enumerate(src_list):
            for j, val in enumerate(val_i):
                src_data[i][j] = val
        src = data_descriptor_from_ctypes(src_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 4)(), writable=True)
        # Do the assignment
        execute_unary_single(dst, src, datashape.float64, datashape.float32, self.sqr)
        self.assertEqual(dd_as_py(dst), [[x*x for x in y] for y in src_list] * 4)

if __name__ == '__main__':
    unittest.main()

