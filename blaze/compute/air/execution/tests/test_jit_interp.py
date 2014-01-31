from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
import blaze
from blaze import array
from blaze.compute.ops.ufuncs import add, multiply

import numpy as np

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_expr(ds1, ds2):
    a = array(range(10), dshape=ds1)
    b = array(range(10), dshape=ds2)
    expr = add(a, multiply(a, b))
    return expr

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestJit(unittest.TestCase):

    def test_jit(self):
        expr = make_expr(dshape('10, float32'), dshape('10, float32'))
        result = blaze.eval(expr, strategy='jit')
        expected = blaze.array([ 0,  2,  6, 12, 20, 30, 42, 56, 72, 90])
        self.assertEqual(type(result), blaze.Array)
        self.assertTrue(np.all(result == expected))

    def test_jit_promotion(self):
        expr = make_expr(dshape('10, int32'), dshape('10, float32'))
        result = blaze.eval(expr, strategy='jit')
        expected = blaze.array([ 0,  2,  6, 12, 20, 30, 42, 56, 72, 90],
                               dshape=dshape('10, float64'))
        self.assertEqual(type(result), blaze.Array)
        self.assertTrue(np.all(result == expected))

    def test_jit_scalar(self):
        a = blaze.array(range(10), dshape=dshape('10, int32'))
        b = 10
        expr = add(a, multiply(a, b))
        result = blaze.eval(expr)
        np_a = np.arange(10)
        expected = np_a + np_a * b
        self.assertTrue(np.all(result == expected))


if __name__ == '__main__':
    # TestJit('test_jit').debug()
    unittest.main()
