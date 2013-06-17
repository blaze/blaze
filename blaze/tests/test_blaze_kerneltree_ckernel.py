import unittest
import sys
import ctypes

import blaze
from blaze.blfuncs import BlazeFunc
from blaze.datashape import double, complex128 as c128
from blaze.datadescriptor import (execute_expr_single, dd_as_py)
from blaze.py3help import izip

class TestBlazeKernelTreeCKernel(unittest.TestCase):
    def test_binary_kerneltree(self):
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
        ubck = df._data.kerneltree.unbound_single_ckernel

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
        af = blaze.array([[1,2,3], [4,5,6]],dshape=double)
        bf = blaze.array([2,3,4],dshape=double)
        cf = add(af,bf)
        df = mul(cf,cf)
        lifted_kernel = df._data.kerneltree.fuse().kernel.lift(1, 'C')
        ubck = lifted_kernel.unbound_single_ckernel

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

if __name__ == '__main__':
    unittest.main()

