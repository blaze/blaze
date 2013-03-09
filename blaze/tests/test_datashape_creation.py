import blaze
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, blaze.dshape, None)
        self.assertRaises(TypeError, blaze.dshape, lambda x: x+1)

    def test_atom_shapes(self):
        self.assertRaises(TypeError, blaze.dshape, 'boot')
        self.assertRaises(TypeError, blaze.dshape, 'int33')

    def test_type_decl(self):
        self.assertRaises(TypeError, blaze.dshape, 'type X T = 3, T')
        self.assertEqual(blaze.dshape('3, int32'), blaze.dshape('type X = 3, int32'))
if __name__ == '__main__':
    unittest.main()
