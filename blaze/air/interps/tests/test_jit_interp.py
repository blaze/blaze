# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

import blaze
from blaze import array, dshape
from blaze.ops.ufuncs import add, mul

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def make_expr(ds1, ds2):
    a = array(range(10), dshape=ds1)
    b = array(range(10), dshape=ds2)
    expr = add(a, mul(a, b))
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
        self.assertEqual(result, expected)

    def test_jit_promotion(self):
        expr = make_expr(dshape('10, int32'), dshape('10, float32'))
        result = blaze.eval(expr, strategy='jit')
        expected = blaze.array([ 0,  2,  6, 12, 20, 30, 42, 56, 72, 90],
                               dshape=dshape('10, float64'))
        self.assertEqual(type(result), blaze.Array)
        self.assertEqual(result, expected)



if __name__ == '__main__':
    # TestJit('test_jit').debug()
    unittest.main()