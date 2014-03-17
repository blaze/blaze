from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from dynd import nd, ndt
import blaze


class evalTest(unittest.TestCase):

    N = 1000
    
    def test00(self):
        """Testing elwise_eval() with only scalars and constants"""
        a = 3
        cr = blaze.elwise_eval("2 * a")
        self.assert_(cr == 6, "eval does not work correctly")

    def test01(self):
        """Testing eval() with only blaze arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a)
        d = blaze.array(b)
        cr = blaze.elwise_eval("c * d")
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing eval() with only dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = nd.array(a)
        d = nd.array(b)
        cr = blaze.elwise_eval("a * b")
        nr = a * b
        #print("blaze.elwise_eval ->", cr)
        #print("numpy   ->", nr)
        assert_array_equal(cr[:], nr, "eval does not work correctly")


if __name__ == '__main__':
    unittest.main()
