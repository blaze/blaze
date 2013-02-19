# -*- coding: utf-8 -*-
"""
Unit tests related to the handling of arrays of objects
-------------------------------------------------------

Notes on object handling:

1. Only one dimensional arrays of objects are handled
2. Composite dtypes that contains objects are currently not handled.

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
        """Testing carray of objects, 2d source
        Expected result will be a 1d carray whose elements are containers
        holding the inner dimension
        """
        src_data = [(i, 's'*i) for i in range(10)]
        carr = ca.carray(src_data, dtype=np.dtype('O'))
        # note that carray should always create a 1 dimensional
        # array of objects.
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_carray_tuple_source(self):
        """Testing a carray of objects that are tuples
        This uses a numpy container as source. Tuples should be
        preserved
        """
        src_data = np.empty((10,), dtype=np.dtype('O'))
        src_data[:] = [(i, 's'*i) for i in range(src_data.shape[0])]
        carr = ca.carray(src_data)
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        self.assertEqual(type(carr[0]), tuple)
        self.assertEqual(type(carr[0]), type(src_data[0]))
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_carray_record(self):
        """Testing carray handling of record dtypes containing objects.
        They must raise a type error exception, as they are not supported
        """
        src_data = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, ca.carray, src_data, dtype=np.dtype('O,O'))

    def test_carray_record_as_object(self):
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        carr = ca.carray(src_data, dtype=np.dtype('O'))
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])
                              
    # The following tests document different alternatives in handling input data
    # which would infer a record dtype in the resulting carray.
    # option 1: fail with a type error as if the dtype was explicit
    # option 2: handle it as an array of arrays of objects.
    def test_carray_record_inferred_opt1(self):
        """Testing carray handling of inferred record dtypes containing objects.
        When there is no explicit dtype in the carray constructor, the dtype is
        inferred. This test checks that an inferred dtype results in a type 
        error.
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, ca.carray, src_data)

    @skipif(True, 'Currently the other option is implemented')
    def test_carray_record_inferred_opt2(self):
        """Testing carray handling of inferred record dtypes containing objects.
        When there is no explicit dtype in the carray constructor, the dtype
        becomes 'O', and the carrays behaves accordingly (one dimensional)
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]

        carr = ca.carray(src_data)
        # note: this is similar as if it was created with dtype='O'
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])


class ObjectCarrayDiskTest(ObjectCarrayTest):
    disk = True

## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 5
## fill-column: 78
