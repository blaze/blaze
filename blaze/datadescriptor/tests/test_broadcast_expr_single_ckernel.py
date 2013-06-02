import unittest
import sys
import ctypes

import blaze
from blaze import datashape
from blaze import ckernel
from blaze.datadescriptor import (data_descriptor_from_ctypes,
                execute_expr_single, dd_as_py)
from blaze.py3help import izip

class TestBroadcastUnarySingleCKernel(unittest.TestCase):
    def setUp(self):
        # Create a ternary single ckernel
        @ckernel.ExprSingleOperation
        def my_ternary_func(dst_ptr, src_ptr, kdp):
            dst = ctypes.c_double.from_address(dst_ptr)
            class binary_src_arg(ctypes.Structure):
                _fields_ = [('src0', ctypes.POINTER(ctypes.c_double)),
                            ('src1', ctypes.POINTER(ctypes.c_double)),
                            ('src2', ctypes.POINTER(ctypes.c_double))]
            src_args = binary_src_arg.from_address(ctypes.addressof(src_ptr.contents))
            src0 = src_args.src0.contents
            src1 = src_args.src1.contents
            src2 = src_args.src2.contents
            dst.value = (src0.value + 1) * src1.value + src2.value
            #print(src0, src1, src2, '->', dst, '--', hex(dst_ptr))
        # The ctypes callback object is both the function and the owner
        self.muladd = ckernel.wrap_ckernel_func(my_ternary_func, my_ternary_func)

    def test_scalar(self):
        # Set up our data buffers
        src0 = data_descriptor_from_ctypes(ctypes.c_double(1028), writable=False)
        src1 = data_descriptor_from_ctypes(ctypes.c_double(5), writable=False)
        src2 = data_descriptor_from_ctypes(ctypes.c_double(-123), writable=False)
        dst = data_descriptor_from_ctypes(ctypes.c_double(-100), writable=True)
        # Do the assignment
        execute_expr_single(dst, [src0, src1, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), (1028.0 + 1) * 5 - 123)

    def test_scalar_to_one_d(self):
        # Set up our data buffers
        src0 = data_descriptor_from_ctypes(ctypes.c_double(1028), writable=False)
        src1 = data_descriptor_from_ctypes(ctypes.c_double(5), writable=False)
        src2 = data_descriptor_from_ctypes(ctypes.c_double(-123), writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3)(), writable=True)
        # Do the assignment
        execute_expr_single(dst, [src0, src1, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(1028.0 + 1) * 5 - 123] * 3)

    def test_broadcast_to_one_d(self):
        # Set up our data buffers
        src0 = data_descriptor_from_ctypes(ctypes.c_double(12), writable=False)
        src1_data = (ctypes.c_double * 1)()
        src1_data[0] = 3
        src1 = data_descriptor_from_ctypes(src1_data, writable=False)
        src2_data = (ctypes.c_double * 3)()
        src2_list = [5, 3, -2]
        for i, val in enumerate(src2_list):
            src2_data[i] = val
        src2 = data_descriptor_from_ctypes(src2_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3)(), writable=True)

        # Do assignments with the different permuations of the source arguments
        execute_expr_single(dst, [src0, src1, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(12 + 1) * 3 + x for x in [5, 3, -2]])
        execute_expr_single(dst, [src0, src2, src1],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(12 + 1) * x + 3 for x in [5, 3, -2]])
        execute_expr_single(dst, [src1, src0, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(3 + 1) * 12 + x for x in [5, 3, -2]])
        execute_expr_single(dst, [src1, src2, src0],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(3 + 1) * x + 12 for x in [5, 3, -2]])
        execute_expr_single(dst, [src2, src0, src1],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(x + 1) * 12 + 3 for x in [5, 3, -2]])
        execute_expr_single(dst, [src2, src1, src0],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [(x + 1) * 3 + 12 for x in [5, 3, -2]])

    def test_broadcast_to_two_d(self):
        # Set up our data buffers
        src0 = data_descriptor_from_ctypes(ctypes.c_double(12), writable=False)
        src0_broadcast = [[12] * 3] * 2
        src1_data = (ctypes.c_double * 3)()
        src1_list = [3, 9, 1]
        src1_broadcast = [src1_list] * 2
        for i, val in enumerate(src1_list):
            src1_data[i] = val
        src1 = data_descriptor_from_ctypes(src1_data, writable=False)
        src2_data = (ctypes.c_double * 3 * 2)()
        src2_list = [[5, 3, -2], [-1, 4, 9]]
        src2_broadcast = src2_list
        for j, val_j in enumerate(src2_list):
            for i, val in enumerate(val_j):
                src2_data[j][i] = val
        src2 = data_descriptor_from_ctypes(src2_data, writable=False)
        dst = data_descriptor_from_ctypes((ctypes.c_double * 3 * 2)(), writable=True)

        # Do assignments with the different permuations of the source arguments
        execute_expr_single(dst, [src0, src1, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for x, y, z in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])
        execute_expr_single(dst, [src0, src2, src1],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for x, z, y in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])
        execute_expr_single(dst, [src1, src0, src2],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for y, x, z in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])
        execute_expr_single(dst, [src1, src2, src0],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for z, x, y in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])
        execute_expr_single(dst, [src2, src0, src1],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for y, z, x in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])
        execute_expr_single(dst, [src2, src1, src0],
                        datashape.float64, [datashape.float64]*3, self.muladd)
        self.assertEqual(dd_as_py(dst), [[(x + 1) * y + z for z, y, x in izip(*tmp)]
                        for tmp in izip(src0_broadcast, src1_broadcast, src2_broadcast)])

if __name__ == '__main__':
    unittest.main()

