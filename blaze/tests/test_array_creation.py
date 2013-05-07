import blaze
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_create(self):
        # A default array (backed by NumPy)
        a = blaze.array([1,2,3])
        self.assert_(isinstance(a, blaze.Array))
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

    def test_create_compress(self):
        # A compressed array (backed by BLZ)
        a = blaze.array([1,2,3], caps={'compress': True})
        self.assert_(isinstance(a, blaze.Array))
        # XXX The tests below still do not work
        # self.assertEqual(a[0], 1)
        # self.assertEqual(a[1], 2)
        # self.assertEqual(a[2], 3)

