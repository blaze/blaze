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

from blaze.optional_packages import tables_is_here
if tables_is_here:
    import tables


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
            os.rmdir(path)
            table = blz.fromiter(
                ((i, i*2.) for i in xrange(self.N)), dtype=self.dtype,
                count=self.N, rootdir=path)
            self.ddesc = blaze.BLZ_DDesc(table, mode='r')
        elif self.disk == 'HDF5' and tables_is_here:
            prefix = 'hdf5-' + self.__class__.__name__
            suffix = '.hdf5'
            dpath = "/table"
            h, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)  # close the not needed file handle
            with tables.open_file(path, "w") as h5f:
                ra = np.fromiter(
                    ((i, i*2.) for i in xrange(self.N)), dtype=self.dtype,
                    count=self.N)
                h5f.create_table('/', dpath[1:], ra)
            self.ddesc = blaze.HDF5_DDesc(path, dpath, mode='r')
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
        nr = self.npt[self.npt['f0'] < 10]
        assert_array_equal(cr, nr, "where does not work correctly")

# Check for tables on-disk (BLZ)
class whereBLZDiskTest(whereTest):
    N = 1000
    disk = "BLZ"

# Check for tables on-disk (HDF5)
class whereHDF5DiskTest(whereTest):
    N = 1000
    disk = "HDF5"



if __name__ == '__main__':
    unittest.main()
