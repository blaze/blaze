from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from datashape import dshape
import blaze
from blaze.compute.function import BlazeFunc
from dynd import nd, _lowlevel


def create_overloaded_add():
    # Create an overloaded blaze func, populate it with
    # some ckernel implementations extracted from numpy,
    # and test some calls on it.
    myfunc = BlazeFunc('test', 'myfunc')

    # overload int32 -> np.add
    ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                                                (np.int32, np.int32, np.int32),
                                                False)
    myfunc.add_overload("(A... * int32, A... * int32) -> A... * int32", ckd)

    # overload int16 -> np.subtract (so we can see the difference)
    ckd = _lowlevel.ckernel_deferred_from_ufunc(np.subtract,
                                                (np.int16, np.int16, np.int16),
                                                False)
    myfunc.add_overload("(A... * int16, A... * int16) -> A... * int16", ckd)

    return myfunc


class TestBlazeFunctionFromUFunc(unittest.TestCase):
    def test_overload(self):
        myfunc = create_overloaded_add()

        # Test int32 overload -> add
        a = blaze.eval(myfunc(blaze.array([3, 4]), blaze.array([1, 2])))
        self.assertEqual(a.dshape, dshape('2 * int32'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [4, 6])
        # Test int16 overload -> subtract
        a = blaze.eval(myfunc(blaze.array([3, 4], dshape='int16'),
                       blaze.array([1, 2], dshape='int16')))
        self.assertEqual(a.dshape, dshape('2 * int16'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [2, 2])

    def test_overload_coercion(self):
        myfunc = create_overloaded_add()

        # Test type promotion to int32
        a = blaze.eval(myfunc(blaze.array([3, 4], dshape='int16'),
                       blaze.array([1, 2])))
        self.assertEqual(a.dshape, dshape('2 * int32'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [4, 6])
        a = blaze.eval(myfunc(blaze.array([3, 4]),
                       blaze.array([1, 2], dshape='int16')))
        self.assertEqual(a.dshape, dshape('2 * int32'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [4, 6])

        # Test type promotion to int16
        a = blaze.eval(myfunc(blaze.array([3, 4], dshape='int8'),
                       blaze.array([1, 2], dshape='int8')))
        self.assertEqual(a.dshape, dshape('2 * int16'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [2, 2])

    def test_nesting(self):
        myfunc = create_overloaded_add()

        # A little bit of nesting
        a = blaze.eval(myfunc(myfunc(blaze.array([3, 4]), blaze.array([1, 2])),
                              blaze.array([2, 10])))
        self.assertEqual(a.dshape, dshape('2 * int32'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [6, 16])

    def test_nesting_and_coercion(self):
        myfunc = create_overloaded_add()

        # More nesting, with conversions
        a = blaze.eval(myfunc(myfunc(blaze.array([1, 2]),
                                     blaze.array([-2, 10])),
                       myfunc(blaze.array([1, 5], dshape='int16'),
                              blaze.array(3, dshape='int16'))))
        self.assertEqual(a.dshape, dshape('2 * int32'))
        self.assertEqual(nd.as_py(a.ddesc.dynd_arr()), [-3, 14])

    def test_overload_different_argcount(self):
        myfunc = BlazeFunc('test', 'ovld')
        # Two parameter overload
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                                                    (np.int32,) * 3,
                                                    False)
        myfunc.add_overload("(A... * int32, A... * int32) -> A... * int32", ckd)

        # One parameter overload
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.negative,
                                                    (np.int32,) * 2, False)
        myfunc.add_overload("(A... * int16, A... * int16) -> A... * int16", ckd)

        return myfunc


if __name__ == '__main__':
    # TestBlazeKernel('test_kernel').debug()
    unittest.main()
