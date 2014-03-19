from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from dynd import nd, ndt
import blaze
from common import MayBeDiskTest


# Check for arrays that fit in the chunk size
class evalTest(unittest.TestCase):
    vm = "numexpr"  # if numexpr not available, it will fall back to python
    N = 1000

    def test00(self):
        """Testing elwise_eval() with only scalars and constants"""
        a = 3
        cr = blaze._elwise_eval("2 * a", vm=self.vm)
        self.assert_(cr == 6, "eval does not work correctly")

    def test01(self):
        """Testing with only blaze arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a)
        d = blaze.array(b)
        cr = blaze._elwise_eval("c * d", vm=self.vm)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing with only numpy arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        cr = blaze._elwise_eval("a * b", vm=self.vm)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing with only dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = nd.array(a)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d", vm=self.vm)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing with a mix of blaze, numpy and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b)
        d = nd.array(a)
        cr = blaze._elwise_eval("a * b + d", vm=self.vm)
        nr = a * b + d
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing with a mix of scalars and blaze, numpy and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b)
        d = nd.array(a)
        cr = blaze._elwise_eval("a * b + d + 2", vm=self.vm)
        nr = a * b + d + 2
        assert_array_equal(cr[:], nr, "eval does not work correctly")


# Check for arrays that are larger than a chunk
class evalLargeTest(evalTest):
    N = 10000

# Check for arrays that fit in the chunk size
# Using the Python VM (i.e. Blaze machinery) here
class evalPythonTest(evalTest):
    vm = "python"

# Check for arrays that are larger than a chunk
# Using the Python VM (i.e. Blaze machinery) here
class evalPythonLargeTest(evalTest):
    N = 10000
    vm = "python"


# Check for arrays stored on-disk, but fit in a chunk
# Check for arrays that fit in memory
class storageTest(MayBeDiskTest):
    N = 1000
    vm = "numexpr"
    disk = True

    def test00(self):
        """Testing elwise_eval() with only blaze arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, storage=self.store1)
        d = blaze.array(b, storage=self.store2)
        cr = blaze._elwise_eval("c * d", vm=self.vm, storage=self.store3)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01(self):
        """Testing elwise_eval() with blaze arrays and constants"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, storage=self.store1)
        d = blaze.array(b, storage=self.store2)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, storage=self.store3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing elwise_eval() with blaze and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, storage=self.store1)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, storage=self.store3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing elwise_eval() with blaze, dynd and numpy arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, storage=self.store1)
        d = nd.array(b)
        cr = blaze._elwise_eval("a * c + d", vm=self.vm, storage=self.store3)
        nr = a * c + d
        assert_array_equal(cr[:], nr, "eval does not work correctly")


# Check for arrays stored on-disk, but are larger than a chunk
class storageLargeTest(storageTest):
    N = 10000

# Check for arrays stored on-disk, but fit in a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storagePythonTest(storageTest):
    vm = "python"

# Check for arrays stored on-disk, but are larger than a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storageLargeTest(storageTest):
    N = 10000
    vm = "python"


if __name__ == '__main__':
    unittest.main()
