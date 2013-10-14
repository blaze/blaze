# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

import blaze
from blaze.function import kernel
from blaze import dshape, array
from dynd import nd, ndt, _lowlevel
import numpy as np

# f

@kernel('X, Y, float32 -> X, Y, float32 -> X, Y, float32')
def f(a, b):
    return a

@kernel('X, Y, complex64 -> X, Y, complex64 -> X, Y, complex64')
def f(a, b):
    return a

@kernel('X, Y, complex128 -> X, Y, complex128 -> X, Y, complex128')
def f(a, b):
    return a

# g

@kernel('X, Y, float32 -> X, Y, float32 -> X, int32')
def g(a, b):
    return a

@kernel('X, Y, float32 -> ..., float32 -> X, int32')
def g(a, b):
    return a

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestBlazeKernel(unittest.TestCase):

    def test_kernel(self):
        A = array([8, 9], dshape('2, int32'))
        res = f(A, A)
        self.assertEqual(str(res.dshape), '1, 2, float32')
        self.assertEqual(len(res.expr), 2)
        graph, ctx = res.expr
        self.assertEqual(len(graph.args), 2)
        self.assertEqual(len(ctx.constraints), 0)
        self.assertEqual(len(ctx.params), 1)
        # res.view()

class TestBlazeFunctionFromUFunc(unittest.TestCase):

    def test_overload(self):
        # Create an overloaded blaze func, populate it with
        # some ckernel implementations extracted from numpy,
        # and test some calls on it.
        d = blaze.overloading.Dispatcher()
        myfunc = blaze.BlazeFunc(d)
        def myfunc_dummy(x, y): raise NotImplementedError

        # overload int32 -> np.add
        sig = blaze.dshape("A..., int32 -> A..., int32 -> A..., int32")
        d.add_overload(myfunc_dummy, sig, {})
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32), False)
        myfunc.implement(myfunc_dummy, sig, "ckernel", ckd)

        # overload int16 -> np.subtract (so we can see the difference)
        sig = blaze.dshape("A..., int16 -> A..., int16 -> A..., int16")
        d.add_overload(myfunc_dummy, sig, {})
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.subtract,
                        (np.int16, np.int16, np.int16), False)
        myfunc.implement(myfunc_dummy, sig, "ckernel", ckd)

        # int32 overload -> add
        a = blaze.eval(myfunc(blaze.array([3,4]), blaze.array([1,2])))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])
        # int16 overload -> subtract
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int16'),
                        blaze.array([1,2], dshape='int16')))
        self.assertEqual(a.dshape, blaze.dshape('2, int16'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [2, 2])

        # type promotion to int32
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int16'),
                        blaze.array([1,2])))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])
        a = blaze.eval(myfunc(blaze.array([3,4]),
                        blaze.array([1,2], dshape='int16')))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])

        # type promotion to int16
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int8'),
                        blaze.array([1,2], dshape='int8')))
        self.assertEqual(a.dshape, blaze.dshape('2, int16'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [2, 2])

if __name__ == '__main__':
    # TestBlazeKernel('test_kernel').debug()
    unittest.main()
