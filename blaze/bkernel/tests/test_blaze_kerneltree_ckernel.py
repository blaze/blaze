import unittest
import sys
import ctypes

import blaze
from blaze.bkernel import BlazeFunc
from blaze.datashape import double, complex128 as c128
from blaze.datadescriptor import (execute_expr_single, dd_as_py)
from blaze.py2help import izip

if ctypes.sizeof(ctypes.c_void_p) == 4:
    c_intptr_t = ctypes.c_int32
else:
    c_intptr_t = ctypes.c_int64

class TestBlazeKernelTreeCKernel(unittest.TestCase):
    def test_binary_kerneltree_single(self):
        # Create some simple blaze funcs, using Numba
        def _add(a,b):
            return a + b

        def _mul(a,b):
            return a * b
        add = BlazeFunc('add',[('f8(f8,f8)', _add),
                               ('c16(c16,c16)', _add)])
        mul = BlazeFunc('mul', {(double,)*3: _mul,
                                (c128,)*3: _mul})
        # Array data and expression
        af = blaze.array([[1,2,3], [4,5,6]],dshape=double)
        bf = blaze.array([2,3,4],dshape=double)
        cf = add(af,bf)
        df = mul(cf,cf)
        ubck = df._data.kerneltree.make_unbound_ckernel(strided=False)

        # Allocate the result, and run the kernel across the arguments
        result = blaze.zeros(df.dshape)
        args = [arg.arr._data for arg in df._data.args]
        ck = ubck.bind(result._data, args)
        execute_expr_single(result._data, args,
                        df._data.kerneltree.kernel.dshapes[-1],
                        df._data.kerneltree.kernel.dshapes[:-1], ck)
        self.assertEqual(dd_as_py(result._data),
                        [[(a+b) * (a+b) for a, b in izip(a1, b1)]
                                for a1, b1 in izip(
                                    [[1,2,3], [4,5,6]], [[2,3,4]]*2)])

        # Use blaze.eval to evaluate cf and df into concrete arrays
        cf2 = blaze.eval(cf)
        self.assertEqual(dd_as_py(cf2._data),
                        [[(a+b) for a, b in izip(a1, b1)]
                                for a1, b1 in izip(
                                    [[1,2,3], [4,5,6]], [[2,3,4]]*2)])
        df2 = blaze.eval(df)
        self.assertEqual(dd_as_py(df2._data),
                        [[(a+b) * (a+b) for a, b in izip(a1, b1)]
                                for a1, b1 in izip(
                                    [[1,2,3], [4,5,6]], [[2,3,4]]*2)])

    def test_binary_kerneltree_isolated(self):
        # Create a simple blaze func, using Numba
        def _testfunc(a,b):
            return a * a + b

        testfunc = BlazeFunc('testfunc',[('f8(f8,f8)', _testfunc)])
        # Use some dummy array data to create a kernel tree
        # and get both single and strided ckernels out of it
        af = blaze.array(1,dshape=double)
        bf = blaze.array(1,dshape=double)
        cf = testfunc(af,bf)
        ubck_single = cf._data.kerneltree.make_unbound_ckernel(strided=False)
        ubck_strided = cf._data.kerneltree.make_unbound_ckernel(strided=True)
        # The particular function we created has a no-op bind(),
        # because it has no strides or similar parameters to set.
        ck_single = ubck_single.bind(None, None)
        ck_strided = ubck_strided.bind(None, None)

        # Set up some data pointers and strides using ctypes
        adata = (ctypes.c_double * 4)()
        for i, v in enumerate([1.25, 3.125, 6.0, -2.25]):
            adata[i] = v
        bdata = (ctypes.c_double * 4)()
        for i, v in enumerate([1.5, -1.5, 2.75, 9.25]):
            bdata[i] = v
        cdata = (ctypes.c_double * 4)()
        src_ptr_arr = (ctypes.c_void_p * 2)()
        src_ptr_arr[0] = ctypes.c_void_p(ctypes.addressof(adata))
        src_ptr_arr[1] = ctypes.c_void_p(ctypes.addressof(bdata))
        src_strides_arr = (c_intptr_t * 2)()
        src_strides_arr[0] = 8
        src_strides_arr[1] = 8

        # Try a single_ckernel call
        ck_single(ctypes.c_void_p(ctypes.addressof(cdata)+8),
                        src_ptr_arr)
        self.assertEqual(cdata[1], _testfunc(adata[0], bdata[0]))
        # Try a strided_ckernel call
        ck_strided(ctypes.addressof(cdata), 8,
                        src_ptr_arr, src_strides_arr,
                        4)
        self.assertEqual(cdata[0], _testfunc(adata[0], bdata[0]))
        self.assertEqual(cdata[1], _testfunc(adata[1], bdata[1]))
        self.assertEqual(cdata[2], _testfunc(adata[2], bdata[2]))
        self.assertEqual(cdata[3], _testfunc(adata[3], bdata[3]))
        # Try a strided_ckernel call, with a zero stride
        src_strides_arr[0] = 0
        ck_strided(ctypes.addressof(cdata), 8,
                        src_ptr_arr, src_strides_arr,
                        4)
        self.assertEqual(cdata[0], _testfunc(adata[0], bdata[0]))
        self.assertEqual(cdata[1], _testfunc(adata[0], bdata[1]))
        self.assertEqual(cdata[2], _testfunc(adata[0], bdata[2]))
        self.assertEqual(cdata[3], _testfunc(adata[0], bdata[3]))

"""
    def test_binary_kerneltree_lifted(self):
        # Create some simple blaze funcs, using Numba
        def _add(a,b):
            return a + b

        def _mul(a,b):
            return a * b
        add = BlazeFunc('add',[('f8(f8,f8)', _add),
                               ('c16(c16,c16)', _add)])
        mul = BlazeFunc('mul', {(double,)*3: _mul,
                                (c128,)*3: _mul})
        # Array data and expression
        af = blaze.array([[1,2,3], [4,5,6]], dshape=double)
        bf = blaze.array([2,3,4], dshape=double)
        cf = add(af,bf)
        df = mul(cf,cf)
        lifted_kernel = df._data.kerneltree.fuse().kernel.lift(1, 'C')
        ubck = lifted_kernel.make_unbound_ckernel(strided=False)

        # Allocate the result, and run the kernel across the arguments
        result = blaze.zeros(df.dshape)
        args = [arg.arr._data for arg in df._data.args]
        ck = ubck.bind(result._data, args)
        execute_expr_single(result._data, args,
                        result._data.dshape.subarray(-2),
                        [a.dshape.subarray(-2) for a in args],
                        ck)
        self.assertEqual(dd_as_py(result._data),
                        [[(a+b) * (a+b) for a, b in izip(a1, b1)]
                                for a1, b1 in izip(
                                    [[1,2,3], [4,5,6]], [[2,3,4]]*2)])
"""
if __name__ == '__main__':
    unittest.main(verbosity=2)

