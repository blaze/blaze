from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from dynd import nd, ndt
import blaze

import unittest
import tempfile
import os
import glob
import blaze
import blz


# Useful superclass for disk-based tests
class createTables(unittest.TestCase):
    disk = None

    def setUp(self):
        self.dtype = 'i4,f8'
        self.npt = np.fromiter(((i, i*2.) for i in xrange(self.N)),
                               dtype=self.dtype, count=self.N)
        if self.disk == 'BLZ':
            prefix = 'blaze-' + self.__class__.__name__
            suffix = '.blz'
            path = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(self.path)
            self.table = blz.fromiter(
                ((i, i*2.) for i in xrange(self.N)), dtype=self.dtype,
                count=self.N, rootdir=self.path)
            self.ddesc = blaze.BLZ_DDesc(path, mode='r')
        elif self.disk == 'HDF5':
            prefix = 'hdf5-' + self.__class__.__name__
            suffix = '.hdf5'
            dpath = "/earray"
            h, path1 = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)  # close the non needed file handle
            self.ddesc1 = blaze.HDF5_DDesc(path1, dpath, mode='w')
            h, path2 = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)
        else:
            table = blz.fromiter(
                ((i, i*2.) for i in xrange(self.N)), dtype=self.dtype,
                count=self.N)
            self.ddesc = blaze.BLZ_DDesc(table, mode='r')

    def tearDown(self):
        self.ddesc.remove()

# Check for tables in-memory
class whereTest(createTables):
    N = 1000

    def test01(self):
        """Testing with only blaze arrays"""
        t = blaze.array(self.ddesc)
        cr = np.fromiter(blaze._where(t, "f0 < 10"), self.dtype)
        #print("cr:", cr)
        nr = self.npt[self.npt['f0'] < 10]
        assert_array_equal(cr, nr, "where does not work correctly")


if __name__ == '__main__':
    unittest.main()
