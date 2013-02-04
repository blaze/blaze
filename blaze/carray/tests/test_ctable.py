########################################################################
#
#       License: BSD
#       Created: September 1, 2010
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

import sys

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase


import blaze.carray as ca
from common import MayBeDiskTest


class createTest(MayBeDiskTest, TestCase):

    def test00a(self):
        """Testing ctable creation from a tuple of carrays"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing ctable creation from a tuple of lists"""
        t = ca.ctable(([1,2,3],[4,5,6]), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([[1,2,3],[4,5,6]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable creation from a tuple of carrays (single column)"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        self.assertRaises(ValueError, ca.ctable, a, 'f0', rootdir=self.rootdir)

    def test01(self):
        """Testing ctable creation from a tuple of numpy arrays"""
        N = 1e1
        a = np.arange(N, dtype='i4')
        b = np.arange(N, dtype='f8')+1
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        #print "t->", `t`
        ra = np.rec.fromarrays([a,b]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing ctable creation from an structured array"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing ctable creation from large iterator"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8',
                        count=N, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing ctable creation from large iterator (with a hint)"""
        N = 10*1000
        ra = np.fromiter(((i, i*2.) for i in xrange(N)),
                         dtype='i4,f8', count=N)
        t = ca.fromiter(((i, i*2.) for i in xrange(N)),
                        dtype='i4,f8', count=N, rootdir=self.rootdir)
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

class createDiskTest(createTest, TestCase):
    disk = True


class persistentTest(MayBeDiskTest, TestCase):

    disk = True

    def test00a(self):
        """Testing ctable opening in "r" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='r')
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        self.assertRaises(RuntimeError, t.__setitem__, 1, (0, 0.0))
        self.assertRaises(RuntimeError, t.append, (0, 0.0))

    def test00b(self):
        """Testing ctable opening in "w" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='w')
        #print "t->", `t`
        N = 0
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((0, 0.0))
        t.append((0, 0.0))
        t[1] = (1, 2.0)
        ra = np.rec.fromarrays([(0,1),(0.0, 2.0)], 'i4,f8').view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00c(self):
        """Testing ctable opening in "a" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Open t
        t = ca.open(rootdir=self.rootdir, mode='a')
        #print "t->", `t`

        # Check values
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[-1] = (11, 12.0)

        # Check values
        N = 12
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing ctable creation in "r" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        self.assertRaises(RuntimeError, ca.ctable, (a, b), ('f0', 'f1'),
                          rootdir=self.rootdir, mode='r')

    def test01b(self):
        """Testing ctable creation in "w" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Overwrite the last ctable
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir, mode='w')
        #print "t->", `t`
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

        # Now check some accesses
        t.append((10, 11.0))
        t.append((10, 11.0))
        t[11] = (11, 12.0)

        # Check values
        N = 12
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        ra = np.rec.fromarrays([a[:],b[:]]).view(np.ndarray)
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01c(self):
        """Testing ctable creation in "a" mode"""
        N = 1e1
        a = ca.carray(np.arange(N, dtype='i4'))
        b = ca.carray(np.arange(N, dtype='f8')+1)
        t = ca.ctable((a, b), ('f0', 'f1'), rootdir=self.rootdir)
        # Overwrite the last ctable
        self.assertRaises(RuntimeError, ca.ctable, (a, b), ('f0', 'f1'),
                          rootdir=self.rootdir, mode='a')


class add_del_colTest(MayBeDiskTest, TestCase):

    def test00a(self):
        """Testing adding a new column (list flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c.tolist(), 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00(self):
        """Testing adding a new column (carray flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(ca.carray(c), 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01a(self):
        """Testing adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01b(self):
        """Testing cparams when adding a new column (numpy flavor)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, cparams=ca.cparams(1), rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, 'f2')
        self.assert_(t['f2'].cparams.clevel == 1, "Incorrect clevel")

    def test02(self):
        """Testing adding a new column (default naming)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(ca.carray(c))
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing inserting a new column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, name='c0', pos=0)
        ra = np.fromiter(((i*3, i, i*2.) for i in xrange(N)), dtype='i8,i4,f8')
        ra.dtype.names = ('c0', 'f0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing inserting a new column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        c = np.arange(N, dtype='i8')*3
        t.addcol(c, name='c0', pos=1)
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        ra.dtype.names = ('f0', 'c0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test05(self):
        """Testing removing an existing column (at the beginning)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=0)
        # The next gives a segfault.  See:
        # http://projects.scipy.org/numpy/ticket/1598
        #ra = np.fromiter(((i*3, i*2) for i in xrange(N)), dtype='i8,f8')
        #ra.dtype.names = ('f1', 'f2')
        dt = np.dtype([('f1', 'i8'), ('f2', 'f8')])
        ra = np.fromiter(((i*3, i*2) for i in xrange(N)), dtype=dt)
        #print "t->", `t`
        #print "ra", ra
        #assert_array_equal(t[:], ra, "ctable values are not correct")

    def test06(self):
        """Testing removing an existing column (at the end)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=2)
        ra = np.fromiter(((i, i*3) for i in xrange(N)), dtype='i4,i8')
        ra.dtype.names = ('f0', 'f1')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test07(self):
        """Testing removing an existing column (in the middle)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.delcol(pos=1)
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test08(self):
        """Testing removing an existing column (by name)"""
        N = 10
        ra = np.fromiter(((i, i*3, i*2.) for i in xrange(N)), dtype='i4,i8,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.delcol('f1')
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        ra.dtype.names = ('f0', 'f2')
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[:], ra, "ctable values are not correct")

class add_del_colDiskTest(add_del_colTest, TestCase):
    disk = True


class getitemTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing __getitem__ with only a start"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start = 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start], ra[start], "ctable values are not correct")

    def test01(self):
        """Testing __getitem__ with start, stop"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start, stop = 3, 9
        #print "t->", `t`
        #print "ra[:]", ra[:]
        assert_array_equal(t[start:stop], ra[start:stop],
                           "ctable values are not correct")

    def test02(self):
        """Testing __getitem__ with start, stop, step"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        start, stop, step = 3, 9, 2
        #print "t->", `t[start:stop:step]`
        #print "ra->", ra[start:stop:step]
        assert_array_equal(t[start:stop:step], ra[start:stop:step],
                           "ctable values are not correct")

    def test03(self):
        """Testing __getitem__ with a column name"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        colname = "f1"
        #print "t->", `t[colname]`
        #print "ra->", ra[colname]
        assert_array_equal(t[colname][:], ra[colname],
                           "ctable values are not correct")

    def test04(self):
        """Testing __getitem__ with a list of column names"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        colnames = ["f0", "f2"]
        # For some version of NumPy (> 1.7) I cannot make use of
        # ra[colnames]   :-/
        ra2 = np.fromiter(((i, i*3) for i in xrange(N)), dtype='i4,i8')
        ra2.dtype.names = ('f0', 'f2')
        #print "t->", `t[colnames]`
        #print "ra2->", ra2
        assert_array_equal(t[colnames][:], ra2,
                           "ctable values are not correct")

class getitemDiskTest(getitemTest, TestCase):
    disk = True


class setitemTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing __setitem__ with only a start"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing __setitem__ with only a stop"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(None, 9, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing __setitem__ with a start, stop"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1,90, None)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing __setitem__ with a start, stop, step"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1,90, 2)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing __setitem__ with a large step"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, chunklen=10, rootdir=self.rootdir)
        sl = slice(1,43, 20)
        t[sl] = (0, 1)
        ra[sl] = (0, 1)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

class setitemDiskTest(setitemTest, TestCase):
    disk = True


class appendTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing append() with scalar values"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t.append((N, N*2))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+1)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing append() with numpy arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((a, b))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing append() with carrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t.append((ca.carray(a), ca.carray(b)))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03(self):
        """Testing append() with structured arrays"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t.append(ra2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test04(self):
        """Testing append() with another ctable"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        ra2 = np.fromiter(((i, i*2.) for i in xrange(N, N+10)), dtype='i4,f8')
        t2 = ca.ctable(ra2)
        t.append(t2)
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        assert_array_equal(t[:], ra, "ctable values are not correct")

class appendDiskTest(appendTest, TestCase):
    disk = True


class trimTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing trim() with Python scalar values"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N-2)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                       rootdir=self.rootdir)
        t.trim(2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing trim() with NumPy scalar values"""
        N = 10000
        ra = np.fromiter(((i, i*2.) for i in xrange(N-200)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.trim(np.int(200))
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02(self):
        """Testing trim() with a complete trim"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(0)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.trim(N)
        self.assert_(len(ra) == len(t), "Lengths are not equal")

class trimDiskTest(trimTest, TestCase):
    disk = True


class resizeTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing resize() (decreasing)"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N-2)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.resize(N-2)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing resize() (increasing)"""
        N = 100
        ra = np.fromiter(((i, i*2.) for i in xrange(N+4)), dtype='i4,f8')
        t = ca.fromiter(((i, i*2.) for i in xrange(N)), 'i4,f8', N,
                        rootdir=self.rootdir)
        t.resize(N+4)
        ra['f0'][N:] = np.zeros(4)
        ra['f1'][N:] = np.zeros(4)
        assert_array_equal(t[:], ra, "ctable values are not correct")

class resizeDiskTest(resizeTest, TestCase):
    disk=True


class copyTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing copy() without params"""
        N = 10
        ra = np.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            rootdir = self.rootdir + "-test00"
        else:
            rootdir = self.rootdir
        t2 = t.copy(rootdir=rootdir, mode='w')
        a = np.arange(N, N+10, dtype='i4')
        b = np.arange(N, N+10, dtype='f8')*2.
        t2.append((a, b))
        ra = np.fromiter(((i, i*2.) for i in xrange(N+10)), dtype='i4,f8')
        self.assert_(len(t) == N, "copy() does not work correctly")
        self.assert_(len(t2) == N+10, "copy() does not work correctly")
        assert_array_equal(t2[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing copy() with higher clevel"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        if self.disk:
            # Copy over the same location should give an error
            self.assertRaises(RuntimeError,
                              t.copy,cparams=ca.cparams(clevel=9),
                              rootdir=self.rootdir, mode='w')
            return
        else:
            t2 = t.copy(cparams=ca.cparams(clevel=9),
                        rootdir=self.rootdir, mode='w')
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t.cparams.clevel == ca.cparams().clevel)
        self.assert_(t2.cparams.clevel == 9)
        self.assert_(t['f1'].cbytes > t2['f1'].cbytes, "clevel not changed")

    def test02(self):
        """Testing copy() with lower clevel"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra, rootdir=self.rootdir)
        t2 = t.copy(cparams=ca.cparams(clevel=1))
        self.assert_(t.cparams.clevel == ca.cparams().clevel)
        self.assert_(t2.cparams.clevel == 1)
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")

    def test03(self):
        """Testing copy() with no shuffle"""
        N = 10*1000
        ra = np.fromiter(((i, i**2.2) for i in xrange(N)), dtype='i4,f8')
        t = ca.ctable(ra)
        # print "t:", t, t.rootdir
        t2 = t.copy(cparams=ca.cparams(shuffle=False), rootdir=self.rootdir)
        #print "cbytes in f1, f2:", t['f1'].cbytes, t2['f1'].cbytes
        self.assert_(t['f1'].cbytes < t2['f1'].cbytes, "clevel not changed")

class copyDiskTest(copyTest, TestCase):
    disk = True


class specialTest(TestCase):

    def test00(self):
        """Testing __len__()"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        self.assert_(len(t) == len(ra), "Objects do not have the same length")

    def test01(self):
        """Testing __sizeof__() (big ctables)"""
        N = int(1e4)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        #print "size t uncompressed ->", t.nbytes
        #print "size t compressed   ->", t.cbytes
        self.assert_(sys.getsizeof(t) < t.nbytes,
                     "ctable does not seem to compress at all")

    def test02(self):
        """Testing __sizeof__() (small ctables)"""
        N = int(111)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        #print "size t uncompressed ->", t.nbytes
        #print "size t compressed   ->", t.cbytes
        self.assert_(sys.getsizeof(t) > t.nbytes,
                     "ctable compress too much??")

class fancy_indexing_getitemTest(TestCase):

    def test00(self):
        """Testing fancy indexing with a small list"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[3,1]]
        rar = ra[[3,1]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing with a large numpy array"""
        N = 10*1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        idx = np.random.randint(1000, size=1000)
        rt = t[idx]
        rar = ra[idx]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test02(self):
        """Testing fancy indexing with an empty list"""
        N = 10*1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[]]
        rar = ra[[]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test03(self):
        """Testing fancy indexing (list of floats)"""
        N = 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra)
        rt = t[[2.3, 5.6]]
        rar = ra[[2.3, 5.6]]
        #print "rt->", rt
        #print "rar->", rar
        assert_array_equal(rt, rar, "ctable values are not correct")

    def test04(self):
        """Testing fancy indexing (list of floats, numpy)"""
        a = np.arange(1,101)
        b = ca.carray(a)
        idx = np.array([1.1, 3.3], dtype='f8')
        self.assertRaises(IndexError, b.__getitem__, idx)


class fancy_indexing_setitemTest(TestCase):

    def test00a(self):
        """Testing fancy indexing (setitem) with a small list"""
        N = 100
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = [3,1]
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test00b(self):
        """Testing fancy indexing (setitem) with a small list (II)"""
        N = 100
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = [3,1]
        t[sl] = [(-1, -2, -3), (-3, -2, -1)]
        ra[sl] = [(-1, -2, -3), (-3, -2, -1)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test01(self):
        """Testing fancy indexing (setitem) with a large array"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(N, size=100)
        t[sl] = (-1, -2, -3)
        ra[sl] = (-1, -2, -3)
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02a(self):
        """Testing fancy indexing (setitem) with a boolean array (I)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(2, size=1000).astype('bool')
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test02b(self):
        """Testing fancy indexing (setitem) with a boolean array (II)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.random.randint(10, size=1000).astype('bool')
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03a(self):
        """Testing fancy indexing (setitem) with a boolean array (all false)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.zeros(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

    def test03b(self):
        """Testing fancy indexing (setitem) with a boolean array (all true)"""
        N = 1000
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=10)
        sl = np.ones(N, dtype="bool")
        t[sl] = [(-1, -2, -3)]
        ra[sl] = [(-1, -2, -3)]
        #print "t[%s] -> %r" % (sl, t)
        #print "ra[%s] -> %r" % (sl, ra)
        assert_array_equal(t[:], ra, "ctable values are not correct")

class iterTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing ctable.__iter__"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test01(self):
        """Testing ctable.iter() without params"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter()]
        nl = [r['f1'] for r in ra]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test02(self):
        """Testing ctable.iter() with start,stop,step"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,3)]
        nl = [r['f1'] for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test03(self):
        """Testing ctable.iter() with outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [tuple(r) for r in t.iter(outcols='f2, nrow__, f0')]
        nl = [(r['f2'], i, r['f0']) for i, r in enumerate(ra)]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test04(self):
        """Testing ctable.iter() with start,stop,step and outcols"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r for r in t.iter(1,9,3, 'f2, nrow__ f0')]
        nl = [(r['f2'], r['f0'], r['f0']) for r in ra[1:9:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test05(self):
        """Testing ctable.iter() with start, stop, step and limit"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, limit=3)]
        nl = [r['f1'] for r in ra[1:9:2][:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test06(self):
        """Testing ctable.iter() with start, stop, step and skip"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, skip=3)]
        nl = [r['f1'] for r in ra[1:9:2][3:]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

    def test07(self):
        """Testing ctable.iter() with start, stop, step and limit, skip"""
        N = 10
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = ca.ctable(ra, chunklen=4, rootdir=self.rootdir)
        cl = [r.f1 for r in t.iter(1,9,2, limit=2, skip=1)]
        nl = [r['f1'] for r in ra[1:9:2][1:3]]
        #print "cl ->", cl
        #print "nl ->", nl
        self.assert_(cl == nl, "iter not working correctily")

class iterDiskTest(iterTest, TestCase):
    disk = True

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
