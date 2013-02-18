# -*- coding: utf-8 -*-
"""
Unit tests related to the handling of arrays of objects
"""

from unittest import TestCase
from numpy.testing.decorators import skipif, knownfailureif

import numpy as np
import blaze.carray as ca
from common import MayBeDiskTest



class ObjectCarrayTest(MayBeDiskTest, TestCase):
    def test_carray_1d_source(self):
        """Testing carray of objects, 1d source"""
        src_data = ['s'*i for i in range(10)]
        carr = ca.carray(src_data, dtype=np.dtype('O'))

        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i], src_data[i])
            self.assertEqual(carr[i], src_data[i])

    def test_carray_2d_source(self):
        """Testing carray of objects, 2d source"""
        src_data = [(i, 's'*i) for i in range(10)]
        carr = ca.carray(src_data, dtype=np.dtype('O'))
        # note that carray should alwas create a 1 dimensional
        # array of objects.
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

#    @knownfailureif(True, 'This coredumps right now')
    def test_carray_1d_composite(self):
        """Testing carray handling of composite dtypes containing objects.
        They must raise a type error exception, as they are not supported
        """
        src_data = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, ca.carray, src_data, dtype=np.dtype('O,O'))


class ObjectCarrayDiskTest(ObjectCarrayTest):
    disk = True

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 5
## fill-column: 78
