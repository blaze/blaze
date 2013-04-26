
import sys
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from blaze import blz
import common


class with_listTest:

    def test00a(self):
        """Testing wheretrue() in combination with a list constructor"""
        a = blz.zeros(self.N, dtype="bool")
        a[30:40] = blz.ones(10, dtype="bool")
        alist = list(a)
        blist1 = [r for r in a.wheretrue()]
        self.assert_(blist1 == range(30,40))
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
        self.assert_(blist1 == range(30,40))
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
        self.assert_(blist1 == range(3,10))
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
        print "l:", l
        self.assert_(l == N - 1)
        self.assert_(s == (N - 1) * (N / 2))  # Gauss summation formula
        


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
