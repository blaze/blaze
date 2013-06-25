from __future__ import print_function, absolute_import

import blaze
from blaze.datadescriptor import dd_as_py
import numpy as np
import unittest
from .common import MayBeUriTest
from blaze.eval import append


class TestEphemeral(unittest.TestCase):

    def test_create_from_numpy(self):
        a = blaze.array(np.arange(3))
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_create(self):
        # A default array (backed by NumPy)
        a = blaze.array([1,2,3])
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3])

    def test_create_append(self):
        # A default array (backed by NumPy, append not supported yet)
        a = blaze.array([])
        self.assert_(isinstance(a, blaze.Array))
        self.assertRaises(NotImplementedError, append, a, [1,2,3])
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_compress(self):
        # A compressed array (backed by BLZ)
        a = blaze.array(np.arange(1,4), caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3])
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_iter(self):
        # A default array (backed by NumPy)
        a = blaze.array((i for i in range(10)))
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), list(range(10)))

    def test_create_compress_iter(self):
        # A compressed array (backed by BLZ)
        a = blaze.array((i for i in range(10)), caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), list(range(10)))

    def test_create_zeros(self):
        # A default array (backed by NumPy)
        a = blaze.zeros('10, int64')
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0]*10)

    def test_create_compress_zeros(self):
        # A compressed array (backed by BLZ)
        a = blaze.zeros('10, int64', caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0]*10)

    def test_create_ones(self):
        # A default array (backed by NumPy)
        a = blaze.ones('10, int64')
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1]*10)

    def test_create_compress_ones(self):
        # A compressed array (backed by BLZ)
        a = blaze.ones('10, int64', caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1]*10)


class TestPersistent(MayBeUriTest, unittest.TestCase):

    uri = True

    def test_create(self):
        persist = blaze.Storage(self.rooturi)
        a = blaze.array([], 'float64', persist=persist)
        self.assert_(isinstance(a, blaze.Array))
        print("->", a.dshape.shape)
        self.assert_(a.dshape.shape == (0,))
        self.assertEqual(dd_as_py(a._data), [])

    def test_append(self):
        persist = blaze.Storage(self.rooturi)
        a = blaze.zeros('0, float64', persist=persist)
        self.assert_(isinstance(a, blaze.Array))
        append(a,list(range(10)))
        self.assertEqual(dd_as_py(a._data), list(range(10)))

    # Using a 1-dim as the internal dimension
    def test_append2(self):
        persist = blaze.Storage(self.rooturi)
        a = blaze.empty('0, 2, float64', persist=persist)
        self.assert_(isinstance(a, blaze.Array))
        lvals = [[i,i*2] for i in range(10)]
        append(a,lvals)
        self.assertEqual(dd_as_py(a._data), lvals)

    def test_open(self):
        persist = blaze.Storage(self.rooturi)
        a = blaze.ones('0, float64', persist=persist)
        append(a,range(10))
        # Re-open the dataset in URI
        a2 = blaze.open(persist)
        self.assert_(isinstance(a2, blaze.Array))
        self.assertEqual(dd_as_py(a2._data), list(range(10)))


# Be sure to run this as python -m blaze.tests.test_array_creation
#  because of the use of relative imports
if __name__ == '__main__':
    unittest.main(verbosity=2)
