from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from dynd import nd, ndt
import blaze

import unittest
import tempfile
import os, os.path
import glob
import blaze


# Useful superclass for disk-based tests
class MayBePersistentTest(unittest.TestCase):
    disk = None

    def setUp(self):
        if self.disk == 'BLZ':
            prefix = 'blaze-' + self.__class__.__name__
            suffix = '.blz'
            path1 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(path1)
            self.ddesc1 = blaze.BLZ_DDesc(path1, mode='w')
            path2 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(path2)
            self.ddesc2 = blaze.BLZ_DDesc(path2, mode='w')
            path3 = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(path3)
            self.ddesc3 = blaze.BLZ_DDesc(path3, mode='w')
        elif self.disk == 'HDF5':
            prefix = 'hdf5-' + self.__class__.__name__
            suffix = '.hdf5'
            dpath = "/earray"
            h, path1 = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)  # close the non needed file handle
            self.ddesc1 = blaze.HDF5_DDesc(path1, dpath, mode='w')
            h, path2 = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)
            self.ddesc2 = blaze.HDF5_DDesc(path2, dpath, mode='w')
            h, path3 = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)
            self.ddesc3 = blaze.HDF5_DDesc(path3, dpath, mode='w')
        else:
            self.ddesc1 = None
            self.ddesc2 = None
            self.ddesc3 = None

    def tearDown(self):
        if self.disk:
            self.ddesc1.remove()
            self.ddesc2.remove()
            self.ddesc3.remove()


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

    def test06(self):
        """Testing reductions on blaze arrays"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b)
        cr = blaze._elwise_eval("sum(b + 2)", vm=self.vm)
        nr = np.sum(b + 2)
        self.assert_(cr == nr, "eval does not work correctly")


# Check for arrays that fit in the chunk size
# Using the Python VM (i.e. Blaze machinery) here
class evalPythonTest(evalTest):
    vm = "python"

# Check for arrays that are larger than a chunk
class evalLargeTest(evalTest):
    N = 10000

# Check for arrays that are larger than a chunk
# Using the Python VM (i.e. Blaze machinery) here
class evalPythonLargeTest(evalTest):
    N = 10000
    vm = "python"


# Check for arrays stored on-disk, but fit in a chunk
# Check for arrays that fit in memory
class storageTest(MayBePersistentTest):
    N = 1000
    vm = "numexpr"
    disk = "BLZ"

    def test00(self):
        """Testing elwise_eval() with only blaze arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = blaze.array(b, ddesc=self.ddesc2)
        cr = blaze._elwise_eval("c * d", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01(self):
        """Testing elwise_eval() with blaze arrays and constants"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = blaze.array(b, ddesc=self.ddesc2)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing elwise_eval() with blaze and dynd arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing elwise_eval() with blaze, dynd and numpy arrays"""
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = nd.array(b)
        cr = blaze._elwise_eval("a * c + d", vm=self.vm, ddesc=self.ddesc3)
        nr = a * c + d
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing reductions on blaze arrays"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a, b = np.arange(self.N), np.arange(1, self.N+1)
        b = blaze.array(b, ddesc=self.ddesc1)
        cr = blaze._elwise_eval("sum(b + 2)", vm=self.vm, ddesc=self.ddesc3)
        nr = np.sum(b + 2)
        self.assert_(cr == nr, "eval does not work correctly")


# Check for arrays stored on-disk, but fit in a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storagePythonTest(storageTest):
    vm = "python"

# Check for arrays stored on-disk, but are larger than a chunk
class storageLargeTest(storageTest):
    N = 10000

# Check for arrays stored on-disk, but are larger than a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storagePythonLargeTest(storageTest):
    N = 10000
    vm = "python"

# Check for arrays stored on-disk, but fit in a chunk
class storageHDF5Test(storageTest):
    disk = "HDF5"

# Check for arrays stored on-disk, but are larger than a chunk
class storageLargeHDF5Test(storageTest):
    N = 10000
    disk = "HDF5"

####################################
# Multidimensional tests start now
####################################

# Check for arrays that fit in a chunk
class evalMDTest(unittest.TestCase):
    N = 10
    M = 100
    vm = "numexpr"

    def test00(self):
        """Testing elwise_eval() with only blaze arrays"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a)
        d = blaze.array(b)
        cr = blaze._elwise_eval("c * d", vm=self.vm)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01(self):
        """Testing elwise_eval() with blaze arrays and scalars"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a)
        d = blaze.array(b)
        cr = blaze._elwise_eval("c * d + 2", vm=self.vm)
        nr = a * b + 2
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test02(self):
        """Testing elwise_eval() with pure dynd arrays and scalars"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = nd.array(a)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d + 2", vm=self.vm)
        nr = a * b + 2
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing elwise_eval() with blaze and dynd arrays and scalars"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d + 2", vm=self.vm)
        nr = a * b + 2
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing reductions on blaze arrays"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        b = blaze.array(b)
        cr = blaze._elwise_eval("sum(b + 2)", vm=self.vm)
        nr = np.sum(b + 2)
        self.assert_(cr == nr, "eval does not work correctly")

    def test05(self):
        """Testing reductions on blaze arrays and axis=0"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        b = blaze.array(b)
        cr = blaze._elwise_eval("sum(b + 2, axis=0)", vm=self.vm)
        nr = np.sum(b + 2, axis=0)
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test06(self):
        """Testing reductions on blaze arrays and axis=1"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        self.assertRaises(NotImplementedError,
                          blaze._elwise_eval, "sum([[1,2],[3,4]], axis=1)")


# Check for arrays that fit in a chunk
# Using the Python VM (i.e. Blaze machinery) here
class evalPythonMDTest(evalMDTest):
    vm = "python"

# Check for arrays that does not fit in a chunk
class evalLargeMDTest(evalMDTest):
    N = 100
    M = 100

# Check for arrays that does not fit in a chunk, but using python VM
class evalPythonLargeMDTest(evalMDTest):
    N = 100
    M = 100
    vm = "python"

# Check for arrays that fit in a chunk (HDF5)
class evalMDHDF5Test(evalMDTest):
    disk = "HDF5"

# Check for arrays that does not fit in a chunk (HDF5)
class evalLargeMDHDF5Test(evalMDTest):
    N = 100
    M = 100
    disk = "HDF5"


# Check for arrays stored on-disk, but fit in a chunk
# Check for arrays that fit in memory
class storageMDTest(MayBePersistentTest):
    N = 10
    M = 100
    vm = "numexpr"
    disk = "BLZ"

    def test00(self):
        """Testing elwise_eval() with only blaze arrays"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = blaze.array(b, ddesc=self.ddesc2)
        cr = blaze._elwise_eval("c * d", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test01(self):
        """Testing elwise_eval() with blaze arrays and constants"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = blaze.array(b, ddesc=self.ddesc2)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test03(self):
        """Testing elwise_eval() with blaze and dynd arrays"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = nd.array(b)
        cr = blaze._elwise_eval("c * d + 1", vm=self.vm, ddesc=self.ddesc3)
        nr = a * b + 1
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test04(self):
        """Testing elwise_eval() with blaze, dynd and numpy arrays"""
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        c = blaze.array(a, ddesc=self.ddesc1)
        d = nd.array(b)
        cr = blaze._elwise_eval("a * c + d", vm=self.vm, ddesc=self.ddesc3)
        nr = a * c + d
        assert_array_equal(cr[:], nr, "eval does not work correctly")

    def test05(self):
        """Testing reductions on blaze arrays"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        b = blaze.array(b, ddesc=self.ddesc1)
        cr = blaze._elwise_eval("sum(b + 2)", vm=self.vm, ddesc=self.ddesc3)
        nr = np.sum(b + 2)
        self.assert_(cr == nr, "eval does not work correctly")

    def test06(self):
        """Testing reductions on blaze arrays and axis=0"""
        if self.vm == "python":
            # The reductions does not work well using Blaze expressions yet
            return
        a = np.arange(self.N*self.M).reshape(self.N, self.M)
        b = np.arange(1, self.N*self.M+1).reshape(self.N, self.M)
        b = blaze.array(b, ddesc=self.ddesc1)
        cr = blaze._elwise_eval("sum(b, axis=0)",
                                vm=self.vm, ddesc=self.ddesc3)
        nr = np.sum(b, axis=0)
        assert_array_equal(cr, nr, "eval does not work correctly")


# Check for arrays stored on-disk, but fit in a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storagePythonMDTest(storageMDTest):
    vm = "python"

# Check for arrays stored on-disk, but are larger than a chunk
class storageLargeMDTest(storageMDTest):
    N = 500

# Check for arrays stored on-disk, but are larger than a chunk
# Using the Python VM (i.e. Blaze machinery) here
class storagePythonLargeMDTest(storageMDTest):
    N = 500
    vm = "python"

# Check for arrays stored on-disk, but fit in a chunk
class storageMDHDF5Test(storageMDTest):
    disk = "HDF5"

# Check for arrays stored on-disk, but are larger than a chunk
class storageLargeMDHDF5Test(storageMDTest):
    N = 500
    disk = "HDF5"


if __name__ == '__main__':
    unittest.main()
