from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
import blaze
from blaze.compute.function import function, kernel
from blaze import array, py2help
from dynd import nd, _lowlevel
import numpy as np

# f

@function('X, Y, float32 -> X, Y, float32 -> X, Y, float32')
def f(a, b):
    return a

@function('X, Y, complex64 -> X, Y, complex64 -> X, Y, complex64')
def f(a, b):
    return a

@function('X, Y, complex128 -> X, Y, complex128 -> X, Y, complex128')
def f(a, b):
    return a

# g

@function('X, Y, float32 -> X, Y, float32 -> X, int32')
def g(a, b):
    return a

@function('X, Y, float32 -> ..., float32 -> X, int32')
def g(a, b):
    return a

def create_overloaded_add():
    # Create an overloaded blaze func, populate it with
    # some ckernel implementations extracted from numpy,
    # and test some calls on it.
    #d = blaze.overloading.Dispatcher()
    @function('A -> A -> A')
    def myfunc(x, y):
        raise NotImplementedError

    # overload int32 -> np.add
    ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                  (np.int32, np.int32, np.int32), False)
    kernel(myfunc, "ckernel", ckd,
           "A..., int32 -> A..., int32 -> A..., int32")

    # overload int16 -> np.subtract (so we can see the difference)
    ckd = _lowlevel.ckernel_deferred_from_ufunc(np.subtract,
                    (np.int16, np.int16, np.int16), False)
    kernel(myfunc, "ckernel", ckd,
           "A..., int16 -> A..., int16 -> A..., int16")

    return myfunc

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


    @py2help.skip
    def test_overload(self):
        myfunc = create_overloaded_add()

        # Test int32 overload -> add
        a = blaze.eval(myfunc(blaze.array([3,4]), blaze.array([1,2])))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])
        # Test int16 overload -> subtract
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int16'),
                        blaze.array([1,2], dshape='int16')))
        self.assertEqual(a.dshape, blaze.dshape('2, int16'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [2, 2])

    @py2help.skip
    def test_overload_coercion(self):
        myfunc = create_overloaded_add()

        # Test type promotion to int32
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int16'),
                        blaze.array([1,2])))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])
        a = blaze.eval(myfunc(blaze.array([3,4]),
                        blaze.array([1,2], dshape='int16')))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [4, 6])

        # Test type promotion to int16
        a = blaze.eval(myfunc(blaze.array([3,4], dshape='int8'),
                        blaze.array([1,2], dshape='int8')))
        self.assertEqual(a.dshape, blaze.dshape('2, int16'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [2, 2])

    @py2help.skip
    def test_nesting(self):
        myfunc = create_overloaded_add()

        # A little bit of nesting
        a = blaze.eval(myfunc(myfunc(blaze.array([3,4]), blaze.array([1,2])),
                        blaze.array([2,10])))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [6, 16])

    @py2help.skip
    def test_nesting_and_coercion(self):
        myfunc = create_overloaded_add()

        # More nesting, with conversions
        a = blaze.eval(myfunc(myfunc(blaze.array([1,2]), blaze.array([-2, 10])),
                       myfunc(blaze.array([1, 5], dshape='int16'),
                              blaze.array(3, dshape='int16'))))
        self.assertEqual(a.dshape, blaze.dshape('2, int32'))
        self.assertEqual(nd.as_py(a._data.dynd_arr()), [-3, 14])


if __name__ == '__main__':
    # TestBlazeKernel('test_kernel').debug()
    unittest.main()
