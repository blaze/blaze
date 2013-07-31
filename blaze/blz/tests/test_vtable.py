########################################################################
#
#       License: BSD
#       Created: July 5, 2013
#       Author:  Francesc Alted - francesc@continuum.io
#
########################################################################

from __future__ import absolute_import

import sys

import numpy as np
from numpy.testing import (
    assert_equal, assert_array_equal, assert_array_almost_equal)
from unittest import TestCase

from blaze import blz
from .common import MayBeDiskTest

if sys.version_info >= (3, 0):
    xrange = range


class createTest(MayBeDiskTest, TestCase):

    def test00(self):
        """Testing vtable creation from a tuple of btables (single row)"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8',
                          count=N, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*3.) for i in xrange(N*2)), dtype='i4,f8',
                          count=N*2, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2), rootdir=self.rootdir)
        r = np.array([(1, 3.)], dtype='i4,f8')[0]
        assert_array_equal(vt[N+1], r, "vtable values are not correct")

    def test01a(self):
        """vtable from a collection of equally sized btables"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N, N*2)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*2)), dtype='i4,f8')
        assert_array_equal(vt[:], ra, "vtable values are not correct")

    def test01b(self):
        """vtable from a collection of differently sized btables"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*2)), dtype='i4,f8')
        assert_array_equal(vt[:], ra, "vtable values are not correct")

    def test01c(self):
        """vtable from a collection of differently sized btables"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        t3 = blz.fromiter(((i, i*2.) for i in xrange(N*2, N*3)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2, t3), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*3)), dtype='i4,f8')
        assert_array_equal(vt[:], ra, "vtable values are not correct")

    def test02a(self):
        """vtable with start"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        t3 = blz.fromiter(((i, i*2.) for i in xrange(N*2, N*3)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2, t3), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*3)), dtype='i4,f8')
        assert_array_equal(vt[2:], ra[2:], "vtable values are not correct")

    def test02b(self):
        """vtable with stop"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        t3 = blz.fromiter(((i, i*2.) for i in xrange(N*2, N*3)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2, t3), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*3)), dtype='i4,f8')
        assert_array_equal(vt[:N*3-2], ra[:N*3-2],
                           "vtable values are not correct")

    def test02c(self):
        """vtable with start, stop"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        t3 = blz.fromiter(((i, i*2.) for i in xrange(N*2, N*3)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2, t3), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*3)), dtype='i4,f8')
        assert_array_equal(vt[3:-4], ra[3:-4],
                           "vtable values are not correct")

    def test02d(self):
        """vtable with start, stop, step"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N+1)),
                          dtype='i4,f8', count=N+1, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*2.) for i in xrange(N+1, N*2)),
                          dtype='i4,f8', count=N-1, rootdir=self.rootdir)
        t3 = blz.fromiter(((i, i*2.) for i in xrange(N*2, N*3)),
                          dtype='i4,f8', count=N, rootdir=self.rootdir)
        vt = blz.vtable((t1, t2, t3), rootdir=self.rootdir)
        ra = np.fromiter(((i, i*2.) for i in xrange(N*3)), dtype='i4,f8')
        assert_array_equal(vt[3:-4:3], ra[3:-4:3],
                           "vtable values are not correct")

