

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import blaze.carray as ca
from blaze.carray import chunk
from blaze.carray.tests import common
from common import MayBeDiskTest


class largeCarrayTest(MayBeDiskTest, TestCase):

    disk = True

    def test00(self):
        """Creating an extremely large carray (> 2**32) in memory."""

        cn = ca.zeros(5e9, dtype="i1")
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

    def test01(self):
        """Creating an extremely large carray (> 2**32) on disk."""

        cn = ca.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

    def test02(self):
        """Opening an extremely large carray (> 2**32) on disk."""

        # Create the array on-disk
        cn = ca.zeros(5e9, dtype="i1", rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))
        # Reopen it from disk
        cn = ca.carray(rootdir=self.rootdir)
        self.assertEqual(len(cn), int(5e9))

        # Now check some accesses
        cn[1] = 1
        self.assertEqual(cn[1], 1)
        cn[int(2e9)] = 2
        self.assertEqual(cn[int(2e9)], 2)
        cn[long(3e9)] = 3
        self.assertEqual(cn[long(3e9)], 3)
        cn[-1] = 4
        self.assertEqual(cn[-1], 4)

        self.assertEqual(cn.sum(), 10)

## Local Variables:
## mode: python
## coding: utf-8 
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:

