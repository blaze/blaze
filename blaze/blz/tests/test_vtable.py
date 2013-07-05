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

    def test00a(self):
        """Testing vtable creation from a tuple of btables"""
        N = int(1e1)
        t1 = blz.fromiter(((i, i*2.) for i in xrange(N)), dtype='i4,f8',
                          count=N, rootdir=self.rootdir)
        t2 = blz.fromiter(((i, i*3.) for i in xrange(N*2)), dtype='i4,f8',
                          count=N*2, rootdir=self.rootdir)

        vt = blz.vtable((t1, t2), rootdir=self.rootdir)
        print "vt->", `vt`
        ra = np.array([(1, 3.)], dtype='i4,f8')[0]
        print "vt[N+1]->", vt[N+1]
        assert_array_equal(vt[N+1], ra, "vtable value is not correct")

