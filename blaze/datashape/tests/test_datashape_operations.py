import blaze
from blaze import datashape
import numpy as np
import unittest

class TestDatashapeOperations(unittest.TestCase):
    def test_scalar_subarray(self):
        self.assertEqual(datashape.int32.subarray(0), datashape.int32)
        self.assertRaises(IndexError, datashape.int32.subarray, 1)
        self.assertEqual(datashape.string.subarray(0), datashape.string)
        self.assertRaises(IndexError, datashape.string.subarray, 1)

    def test_array_subarray(self):
        self.assertEqual(datashape.dshape('3, int32').subarray(0),
                        datashape.dshape('3, int32'))
        self.assertEqual(datashape.dshape('3, int32').subarray(1),
                        datashape.int32)
        self.assertEqual(datashape.dshape('3, var, M, int32').subarray(2),
                        datashape.dshape('M, int32'))
        self.assertEqual(datashape.dshape('3, var, M, float64').subarray(3),
                        datashape.float64)

    def test_dshape_compare(self):
        self.assertNotEqual(datashape.int32, datashape.dshape('1, int32'))

if __name__ == '__main__':
    unittest.main()
