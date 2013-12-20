from __future__ import print_function, division, absolute_import

import unittest

from datashape import dshape
import blaze
from blaze import array
from blaze.compute.ops.ufuncs import add, mul

import numpy as np

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestPython(unittest.TestCase):

    def test_interp(self):
        a = array(range(10), dshape=dshape('10, int32'))
        b = array(range(10), dshape=dshape('10, float32'))
        expr = add(a, mul(a, b))
        result = blaze.eval(expr, strategy='py')
        expected = blaze.array([ 0,  2,  6, 12, 20, 30, 42, 56, 72, 90])
        self.assertEqual(type(result), blaze.Array)
        self.assertTrue(np.all(result == expected))


if __name__ == '__main__':
    unittest.main()
