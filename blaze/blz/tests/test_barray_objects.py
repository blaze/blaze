from __future__ import absolute_import

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
from blaze import blz
from .common import MayBeDiskTest

class ObjectBarrayTest(MayBeDiskTest, TestCase):
    def test_barray_1d_source(self):
        """Testing barray of objects, 1d source"""
        src_data = ['s'*i for i in range(10)]
        carr = blz.barray(src_data, dtype=np.dtype('O'))

        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i], src_data[i])
            self.assertEqual(carr[i], src_data[i])

    def test_barray_2d_source(self):
        """Testing barray of objects, 2d source

        Expected result will be a 1d barray whose elements are
        containers holding the inner dimension
        """
        src_data = [(i, 's'*i) for i in range(10)]
        carr = blz.barray(src_data, dtype=np.dtype('O'))
        # note that barray should always create a 1 dimensional
        # array of objects.
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_barray_tuple_source(self):
        """Testing a barray of objects that are tuples

        This uses a numpy container as source. Tuples should be
        preserved
        """
        src_data = np.empty((10,), dtype=np.dtype('O'))
        src_data[:] = [(i, 's'*i) for i in range(src_data.shape[0])]
        carr = blz.barray(src_data)
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        self.assertEqual(type(carr[0]), tuple)
        self.assertEqual(type(carr[0]), type(src_data[0]))
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    def test_barray_record(self):
        """Testing barray handling of record dtypes containing
        objects.  They must raise a type error exception, as they are
        not supported
        """
        src_data = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, blz.barray, src_data, dtype=np.dtype('O,O'))

    def test_barray_record_as_object(self):
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        carr = blz.barray(src_data, dtype=np.dtype('O'))
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])

    # The following tests document different alternatives in handling
    # input data which would infer a record dtype in the resulting
    # barray.
    #
    # option 1: fail with a type error as if the dtype was
    # explicit
    #
    # option 2: handle it as an array of arrays of objects.
    def test_barray_record_inferred_opt1(self):
        """Testing barray handling of inferred record dtypes
        containing objects.  When there is no explicit dtype in the
        barray constructor, the dtype is inferred. This test checks
        that an inferred dtype results in a type error.
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]
        self.assertRaises(TypeError, blz.barray, src_data)

    @skipif(True, 'Currently the other option is implemented')
    def test_barray_record_inferred_opt2(self):
        """Testing barray handling of inferred record dtypes
        containing objects.  When there is no explicit dtype in the
        barray constructor, the dtype becomes 'O', and the barrays
        behaves accordingly (one dimensional)
        """
        src_data = np.empty((10,), dtype=np.dtype('u1,O'))
        src_data[:] = [(i, 's'*i) for i in range(10)]

        carr = blz.barray(src_data)
        # note: this is similar as if it was created with dtype='O'
        self.assertEqual(len(carr.shape), 1)
        self.assertEqual(len(src_data), carr.shape[0])
        for i in range(len(carr)):
            self.assertEqual(carr[i][0], src_data[i][0])
            self.assertEqual(carr[i][1], src_data[i][1])


class ObjectBarrayDiskTest(ObjectBarrayTest):
    disk = True



## Local Variables:
## mode: python
## coding: utf-8
## py-indent-offset: 4
## tab-with: 4
## fill-column: 66
## End:

