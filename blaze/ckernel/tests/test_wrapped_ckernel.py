import unittest
import ctypes

from blaze import ckernel

class TestWrappedCKernel(unittest.TestCase):
    def test_ctypes_callback(self):
        def my_kernel_func(dst_ptr, src_ptr, kdp):
            dst = ctypes.c_int32.from_address(dst_ptr)
            src = ctypes.c_double.from_address(src_ptr)
            dst.value = int(src.value * 3.5)
        my_callback = ckernel.UnarySingleOperation(my_kernel_func)
        # The ctypes callback object is both the function and the owner
        ck = ckernel.wrap_ckernel_func(my_callback, my_callback)
        # Delete the callback to make sure the ckernel is holding a reference
        del my_callback
        # Make some memory and call the kernel
        src_val = ctypes.c_double(4.0)
        dst_val = ctypes.c_int32(-1)
        ck(ctypes.addressof(dst_val), ctypes.addressof(src_val))
        self.assertEqual(dst_val.value, 14)

if __name__ == '__main__':
    unittest.main()

