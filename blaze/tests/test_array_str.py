import unittest
import blaze

class TestArrayStr(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(str(blaze.array(100)), '100')
        self.assertEqual(str(blaze.array(-3.25)), '-3.25')
        self.assertEqual(str(blaze.array(True)), 'True')
        self.assertEqual(str(blaze.array(False)), 'False')

    def test_1d_array(self):
        self.assertEqual(str(blaze.array([1,2,3])), '[1 2 3]')
