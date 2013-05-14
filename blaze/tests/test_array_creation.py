import blaze
from blaze.datadescriptor import dd_as_py
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_create_from_numpy(self):
        a = blaze.array(np.arange(3))
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [0, 1, 2])

    def test_create(self):
        # A default array (backed by NumPy)
        a = blaze.array([1,2,3])
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), [1, 2, 3])
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_compress(self):
        # A compressed array (backed by BLZ)
        a = blaze.array([1,2,3], caps={'compress': True})
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
        self.assertEqual(dd_as_py(a._data), range(10))

    def test_create_compress_iter(self):
        # A compressed array (backed by BLZ)
        a = blaze.array((i for i in range(10)), caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        self.assertEqual(dd_as_py(a._data), range(10))

if __name__ == '__main__':
    unittest.main()
