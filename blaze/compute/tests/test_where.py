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
    open = False

    def setUp(self):
        self.dtype = 'i4,f8'
        self.npt = np.fromiter(((i, i*2.) for i in range(self.N)),
                               dtype=self.dtype, count=self.N)
        if self.disk == 'BLZ':
            prefix = 'blaze-' + self.__class__.__name__
            suffix = '.blz'
            path = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
            os.rmdir(path)
            if self.open:
                table = blz.fromiter(
                    ((i, i*2.) for i in range(self.N)), dtype=self.dtype,
                    count=self.N, rootdir=path)
                self.ddesc = blaze.BLZ_DDesc(table, mode='r')
            else:
                self.ddesc = blaze.BLZ_DDesc(path, mode='w')
                a = blaze.array([(i, i*2.) for i in range(self.N)],
                                'var * {f0: int32, f1: float64}',
                                ddesc=self.ddesc)
        elif self.disk == 'HDF5' and tables_is_here:
            prefix = 'hdf5-' + self.__class__.__name__
            suffix = '.hdf5'
            dpath = "/table"
            h, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
            os.close(h)  # close the not needed file handle
            if self.open:
                with tables.open_file(path, "w") as h5f:
                    ra = np.fromiter(
                        ((i, i*2.) for i in range(self.N)), dtype=self.dtype,
                        count=self.N)
                    h5f.create_table('/', dpath[1:], ra)
                self.ddesc = blaze.HDF5_DDesc(path, dpath, mode='r')
            else:
                self.ddesc = blaze.HDF5_DDesc(path, dpath, mode='w')
                a = blaze.array([(i, i*2.) for i in range(self.N)],
                                'var * {f0: int32, f1: float64}',
                                ddesc=self.ddesc)
        else:
            table = blz.fromiter(
                ((i, i*2.) for i in range(self.N)), dtype=self.dtype,
                count=self.N)
            self.ddesc = blaze.BLZ_DDesc(table, mode='r')

    def tearDown(self):
        self.ddesc.remove()

# Check for tables in-memory (BLZ)
class whereTest(createTables):
    N = 1000

    def test00(self):
        """Testing the dshape attribute of a streamed array"""
        t = blaze.array(self.ddesc)
        st = t.where("f0 < 10")
        self.assert_(isinstance(st, blaze.Array))
        self.assert_(isinstance(st.ddesc, blaze.Stream_DDesc))
        self.assert_(t.dshape.measure == st.dshape.measure)

    def test01(self):
        """Testing with a filter in only one field"""
        t = blaze.array(self.ddesc)
        st = t.where("f0 < 10")
        cr = [tuple(i.values()) for i in st]
        nr = [tuple(i) for i in self.npt[self.npt['f0'] < 10]]
        #print("cr:", cr)
        #print("nr:", nr)
        self.assert_(cr == nr, "where does not work correctly")

    def test02(self):
        """Testing with two fields"""
        t = blaze.array(self.ddesc)
        st = t.where("(f0 < 10) & (f1 > 4)")
        cr = [tuple(i.values()) for i in st]
        nr = [tuple(i) for i in self.npt[
            (self.npt['f0'] < 10) & (self.npt['f1'] > 4)]]
        self.assert_(cr == nr, "where does not work correctly")

# Check for tables on-disk (BLZ)
class whereBLZDiskTest(whereTest):
    disk = "BLZ"

# Check for tables on-disk (HDF5)
class whereHDF5DiskTest(whereTest):
    disk = "HDF5"

# Check for tables on-disk, using existing BLZ files
class whereBLZDiskOpenTest(whereTest):
    disk = "BLZ"
    open = True

# Check for tables on-disk, using existng HDF5 files
class whereHDF5DiskOpenTest(whereTest):
    disk = "HDF5"
    open = True



if __name__ == '__main__':
    unittest.main()
