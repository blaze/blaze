import unittest
import sys
import ctypes

import blaze
from blaze import datashape
from blaze import ckernel
from blaze.datadescriptor import (data_descriptor_from_ctypes,
                execute_expr_single, dd_as_py)

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
            dst.value = src0.value * src1.value + src2.value
            #print(src0, src1, src2, '->', dst)
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
        self.assertEqual(dd_as_py(dst), 1028.0 * 5 - 123)

if __name__ == '__main__':
    unittest.main()

