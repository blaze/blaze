import unittest
import ctypes
import sys

from blaze.compute import ckernel
from blaze.py2help import skipIf
from dynd import nd, ndt, _lowlevel

# On 64-bit windows python 2.6 appears to have
# ctypes bugs in the C calling convention, so
# disable these tests.
win64_py26 = (sys.platform == 'win32' and
              ctypes.sizeof(ctypes.c_void_p) == 8 and
              sys.version_info[:2] <= (2, 6))

class TestWrappedCKernel(unittest.TestCase):
    @skipIf(win64_py26, 'py26 win64 ctypes is buggy')
    def test_ctypes_callback(self):
        # Create a ckernel directly with ctypes
        def my_kernel_func(dst_ptr, src_ptr, kdp):
            dst = ctypes.c_int32.from_address(dst_ptr)
            src = ctypes.c_double.from_address(src_ptr)
            dst.value = int(src.value * 3.5)
        my_callback = _lowlevel.UnarySingleOperation(my_kernel_func)
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            # The ctypes callback object is both the function and the owner
            ckernel.wrap_ckernel_func(ckb, 0, my_callback, my_callback)
            # Delete the callback to make sure the ckernel is holding a reference
            del my_callback
            # Make some memory and call the kernel
            src_val = ctypes.c_double(4.0)
            dst_val = ctypes.c_int32(-1)
            ck = ckb.ckernel(_lowlevel.UnarySingleOperation)
            ck(ctypes.addressof(dst_val), ctypes.addressof(src_val))
            self.assertEqual(dst_val.value, 14)

    @skipIf(win64_py26, 'py26 win64 ctypes is buggy')
    def test_ctypes_callback_deferred(self):
        # Create a deferred ckernel via a closure
        def instantiate_ckernel(out_ckb, ckb_offset, types, meta, kerntype):
            out_ckb = _lowlevel.CKernelBuilder(out_ckb)
            def my_kernel_func_single(dst_ptr, src_ptr, kdp):
                dst = ctypes.c_int32.from_address(dst_ptr)
                src = ctypes.c_double.from_address(src_ptr[0])
                dst.value = int(src.value * 3.5)
            def my_kernel_func_strided(dst_ptr, dst_stride, src_ptr, src_stride, count, kdp):
                src_ptr0 = src_ptr[0]
                src_stride0 = src_stride[0]
                for i in range(count):
                    my_kernel_func_single(dst_ptr, [src_ptr0], kdp)
                    dst_ptr += dst_stride
                    src_ptr0 += src_stride0
            if kerntype == 'single':
                kfunc = _lowlevel.ExprSingleOperation(my_kernel_func_single)
            else:
                kfunc = _lowlevel.ExprStridedOperation(my_kernel_func_strided)
            return ckernel.wrap_ckernel_func(out_ckb, ckb_offset,
                            kfunc, kfunc)
        ckd = _lowlevel.ckernel_deferred_from_pyfunc(instantiate_ckernel,
                        [ndt.int32, ndt.float64])
        # Test calling the ckd
        out = nd.empty(ndt.int32)
        in0 = nd.array(4.0, type=ndt.float64)
        ckd.__call__(out, in0)
        self.assertEqual(nd.as_py(out), 14)

        # Also call it lifted
        ckd_lifted = _lowlevel.lift_ckernel_deferred(ckd,
                        ['2, var, int32', '2, var, float64'])
        out = nd.empty('2, var, int32')
        in0 = nd.array([[1.0, 3.0, 2.5], [1.25, -1.5]], type='2, var, float64')
        ckd_lifted.__call__(out, in0)
        self.assertEqual(nd.as_py(out), [[3, 10, 8], [4, -5]])

if __name__ == '__main__':
    unittest.main()

