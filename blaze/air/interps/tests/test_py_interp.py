# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

import blaze
from blaze import array, dshape
from blaze import error
from blaze.ops.ufuncs import add, mul
from blaze.air import interps
from blaze.datashape import unify, dshapes, coerce, normalize

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestPython(unittest.TestCase):

    def test_interp(self):
        a = array(range(10), dshape=dshape('10, int32'))
        b = array(range(10), dshape=dshape('10, float32'))
        expr = add(a, mul(a, b))
        result = blaze.eval(expr, strategy='py')


if __name__ == '__main__':
    unittest.main()