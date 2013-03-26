"""
Test for compatability between NumPy and a subset of Blaze types.
"""

import unittest
import blaze
import numpy as np

from blaze import dshape
from blaze.datashape.coretypes import NotNumpyCompatible,\
    to_numpy, from_numpy, extract_dims, extract_measure

from blaze.test_utils import assert_raises

#------------------------------------------------------------------------
# To NumPy
#------------------------------------------------------------------------

class TestToNumPy(unittest.TestCase):
    def test_dtype_compat(self):
        self.assertEqual(to_numpy(blaze.int32), np.dtype(np.int32))
        self.assertEqual(to_numpy(blaze.int64), np.dtype(np.int64))
        self.assertEqual(to_numpy(blaze.float_), np.dtype(np.float_))
        self.assertEqual(to_numpy(blaze.int_), np.dtype(np.int_))

    def test_shape_compat(self):
        self.assertEqual(to_numpy(dshape('1, int32')), ((1,), np.int32))
        self.assertEqual(to_numpy(dshape('1, 2, int32')), ((1, 2), np.int32))
        self.assertEqual(to_numpy(dshape('1, 2, 3, 4, int32')), ((1, 2, 3, 4), np.int32))

    def test_deconstruct(self):
        ds = dshape('1, 2, 3, int32')

        self.assertEqual([int(x) for x in extract_dims(ds)], [1,2,3])
        self.assertEqual(extract_measure(ds), blaze.int32)

    def test_not_compat(self):
        with assert_raises(NotNumpyCompatible):
            to_numpy(dshape('Range(0, 3), int32'))

#------------------------------------------------------------------------
# From NumPy
#------------------------------------------------------------------------

class TestFromNumPy(unittest.TestCase):
    def test_from_numpy(self):
        self.assertEqual(from_numpy((), np.int32), blaze.int32)
        self.assertEqual(from_numpy((), np.int_), blaze.int_)

        self.assertEqual(from_numpy((1,), np.int32), blaze.dshape('1, int32'))
        self.assertEqual(from_numpy((1,2), np.int32), blaze.dshape('1, 2, int32'))
