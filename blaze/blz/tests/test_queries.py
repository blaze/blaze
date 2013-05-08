from __future__ import absolute_import

import sys
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from blaze import blz

if sys.version_info >= (3, 0):
    xrange = range

class with_listTest:

    def test00a(self):
        """Testing wheretrue() in combination with a list constructor"""
        a = blz.zeros(self.N, dtype="bool")
        a[30:40] = blz.ones(10, dtype="bool")
        alist = list(a)
        blist1 = [r for r in a.wheretrue()]
        self.assert_(blist1 == list(range(30,40)))
        alist2 = list(a)
        self.assert_(alist == alist2, "wheretrue() not working correctly")

    def test00b(self):
        """Testing wheretrue() with a multidimensional array"""
        a = blz.zeros((self.N, 10), dtype="bool")
        a[30:40] = blz.ones(10, dtype="bool")
        self.assertRaises(NotImplementedError, a.wheretrue)

    def test01a(self):
        """Testing where() in combination with a list constructor"""
        a = blz.zeros(self.N, dtype="bool")
        a[30:40] = blz.ones(10, dtype="bool")
        b = blz.arange(self.N, dtype="f4")
        blist = list(b)
        blist1 = [r for r in b.where(a)]
        self.assert_(blist1 == list(range(30,40)))
        blist2 = list(b)
        self.assert_(blist == blist2, "where() not working correctly")

    def test01b(self):
        """Testing where() with a multidimensional array"""
        a = blz.zeros((self.N, 10), dtype="bool")
        a[30:40] = blz.ones(10, dtype="bool")
        b = blz.arange(self.N*10, dtype="f4").reshape((self.N, 10))
        self.assertRaises(NotImplementedError, b.where, a)

    def test02(self):
        """Testing iter() in combination with a list constructor"""
        b = blz.arange(self.N, dtype="f4")
        blist = list(b)
        blist1 = [r for r in b.iter(3,10)]
        self.assert_(blist1 == list(range(3,10)))
        blist2 = list(b)
        self.assert_(blist == blist2, "iter() not working correctly")


class small_with_listTest(with_listTest, TestCase):
    N = 100

class big_with_listTest(with_listTest, TestCase):
    N = 10000


class wherechunksTest(TestCase):

    def test00(self):
        """Testing `wherechunks` method with only an expression"""
        N = int(1e4)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2'):
            l += len(block)
            s += block['f0'].sum()
        self.assert_(l == N - 1)
        self.assert_(s == (N - 1) * (N / 2))  # Gauss summation formula

    def test01(self):
        """Testing `wherechunks` method with a `blen`"""
        N = int(1e4)
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f0 <= f1', blen=100):
            l += len(block)
            # All blocks should be of length 100, except the last one,
            # which should be 0
            self.assert_(len(block) in (0, 100))
            s += block['f0'].sum()
        self.assert_(l == N)
        self.assert_(s == (N - 1) * (N / 2))  # Gauss summation formula

    def test02(self):
        """Testing `wherechunks` method with a `outfields` with 2 fields"""
        N = int(1e4)
        ra = np.fromiter(((i, i, i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', outfields=('f1','f2')):
            self.assert_(block.dtype.names == ('f1','f2'))
            l += len(block)
            s += block['f1'].sum()
        self.assert_(l == N - 1)
        self.assert_(s == (N - 1) * (N / 2))  # Gauss summation formula

    def test03(self):
        """Testing `wherechunks` method with a `outfields` with 1 field"""
        N = int(1e4)
        ra = np.fromiter(((i, i, i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', outfields=('f1',)):
            self.assert_(block.dtype.names == ('f1',))
            l += len(block)
            s += block['f1'].sum()
        self.assert_(l == N - 1)
        self.assert_(s == (N - 1) * (N / 2))  # Gauss summation formula

    def test04(self):
        """Testing `wherechunks` method with a `limit` parameter"""
        N, M = int(1e4), 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', limit=M):
            l += len(block)
            s += block['f0'].sum()
        self.assert_(l == M)
        self.assert_(s == M * ((M + 1) / 2))  # Gauss summation formula

    def test05(self):
        """Testing `wherechunks` method with a `limit` parameter"""
        N, M = int(1e4), 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', limit=M):
            l += len(block)
            s += block['f0'].sum()
        self.assert_(l == M)
        self.assert_(s == M * ((M + 1) / 2))  # Gauss summation formula

    def test06(self):
        """Testing `wherechunks` method with a `skip` parameter"""
        N, M = int(1e4), 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', skip=N-M):
            l += len(block)
            s += block['f0'].sum()
        self.assert_(l == M - 1)
        self.assert_(s == np.arange(N-M+1, N).sum())

    def test07(self):
        """Testing `wherechunks` method with a `limit`, `skip` parameter"""
        N, M = int(1e4), 101
        ra = np.fromiter(((i, i*2., i*3) for i in xrange(N)), dtype='i4,f8,i8')
        t = blz.btable(ra)
        l, s = 0, 0
        for block in blz.whereblocks(t, 'f1 < f2', limit=N-M-2, skip=M):
            l += len(block)
            s += block['f0'].sum()
        self.assert_(l == N - M - 2)
        self.assert_(s == np.arange(M+1, N-1).sum())


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
