import blaze
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, blaze.dshape, None)
        self.assertRaises(TypeError, blaze.dshape, lambda x: x+1)

if __name__ == '__main__':
    unittest.main()
