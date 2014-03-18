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
        cr = blaze._elwise_eval("2 * a")
        self.assert_(cr == 6, "eval does not work correctly")

    def test01(self):
        """Testing with only blaze arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a)
        d = blaze.array(b)
        cr = blaze._elwise_eval("c * d")
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing with only numpy arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        cr = blaze._elwise_eval("a * b")
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing with only dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = nd.array(a)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d")
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing with a mix of blaze, numpy and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b)
        d = nd.array(a)
        cr = blaze._elwise_eval("a * b + d")
        nr = a * b + d
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing with a mix of scalars and blaze, numpy and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b)
        d = nd.array(a)
        cr = blaze._elwise_eval("a * b + d + 2")
        nr = a * b + d + 2
        assert_array_equal(cr[:], nr, "eval does not work correctly")

# Check for arrays that fit in memory, but are larger than a chunk
class evalLargeTest(evalTest):
    N = 10000


if __name__ == '__main__':
    unittest.main()
