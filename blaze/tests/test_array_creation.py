from __future__ import absolute_import, division, print_function

import blaze
import datashape
from blaze.datadescriptor import dd_as_py
import numpy as np
import unittest
from blaze.tests.common import MayBeUriTest
from blaze import append
from ..py2help import skip

class TestEphemeral(unittest.TestCase):

    def test_create_from_numpy(self):
        a = blaze.array(np.arange(3))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_create(self):
        # A default array (backed by NumPy)
        a = blaze.array([1,2,3])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3])

    @skip('A NotImplementedError should be raised')
    def test_create_append(self):
        # A default array (backed by NumPy, append not supported yet)
        a = blaze.array([])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertRaises(NotImplementedError, append, a, [1,2,3])
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_compress(self):
        # A compressed array (backed by BLZ)
        a = blaze.array(np.arange(1,4), caps={'compress': True})
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3])
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_iter(self):
        # A simple 1D array
        a = blaze.array(i for i in range(10))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('10, int32'))
        self.assertEqual(dd_as_py(a._data), list(range(10)))
        # A nested iter
        a = blaze.array((i for i in range(x)) for x in range(5))
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('5, var, int32'))
        self.assertEqual(dd_as_py(a._data),
                         [[i for i in range(x)] for x in range(5)])
        # A list of iter
        a = blaze.array([range(3), (1.5*x for x in range(4)), iter([-1, 1])])
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(a.dshape, datashape.dshape('3, var, float64'))
        self.assertEqual(dd_as_py(a._data),
                         [list(range(3)),
                          [1.5*x for x in range(4)],
                          [-1, 1]])

    def test_create_compress_iter(self):
        # A compressed array (backed by BLZ)
        a = blaze.array((i for i in range(10)), caps={'compress': True})
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), list(range(10)))

    def test_create_zeros(self):
        # A default array
        a = blaze.zeros('10, int64')
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0]*10)

    def test_create_compress_zeros(self):
        # A compressed array (backed by BLZ)
        a = blaze.zeros('10, int64', caps={'compress': True})
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0]*10)

    def test_create_ones(self):
        # A default array
        a = blaze.ones('10, int64')
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1]*10)

    def test_create_compress_ones(self):
        # A compressed array (backed by BLZ)
        a = blaze.ones('10, int64', caps={'compress': True})
        self.assertTrue(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1]*10)

    def test_create_record(self):
        # A simple record array
        a = blaze.array([(10, 3.5), (15, 2.25)],
                        dshape="var, {val: int32; flt: float32}")
        self.assertEqual(dd_as_py(a._data), [{'val': 10, 'flt': 3.5},
                        {'val': 15, 'flt': 2.25}])
        # Test field access via attributes
        aval = a.val
        self.assertEqual(dd_as_py(aval._data), [10, 15])
        aflt = a.flt
        self.assertEqual(dd_as_py(aflt._data), [3.5, 2.25])

class TestPersistent(MayBeUriTest, unittest.TestCase):

    uri = True

    def test_create(self):
        persist = blaze.Storage(self.rooturi, format="blz")
        a = blaze.array([], 'float64', storage=persist)
        self.assertTrue(isinstance(a, blaze.Array))
        print("->", a.dshape.shape)
        self.assertTrue(a.dshape.shape == (0,))
        self.assertEqual(dd_as_py(a._data), [])

    def test_append(self):
        persist = blaze.Storage(self.rooturi, format="blz")
        a = blaze.zeros('0, float64', storage=persist)
        self.assertTrue(isinstance(a, blaze.Array))
        append(a,list(range(10)))
        self.assertEqual(dd_as_py(a._data), list(range(10)))

    # Using a 1-dim as the internal dimension
    def test_append2(self):
        persist = blaze.Storage(self.rooturi, format="blz")
        a = blaze.empty('0, 2, float64', storage=persist)
        self.assertTrue(isinstance(a, blaze.Array))
        lvals = [[i,i*2] for i in range(10)]
        append(a,lvals)
        self.assertEqual(dd_as_py(a._data), lvals)

    def test_open(self):
        persist = blaze.Storage(self.rooturi, format="blz")
        a = blaze.ones('0, float64', storage=persist)
        append(a,range(10))
        # Re-open the dataset in URI
        a2 = blaze.open(persist)
        self.assertTrue(isinstance(a2, blaze.Array))
        self.assertEqual(dd_as_py(a2._data), list(range(10)))

if __name__ == '__main__':
    unittest.main(verbosity=2)
