import blaze
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, blaze.dshape, None)
        self.assertRaises(TypeError, blaze.dshape, lambda x: x+1)

    def test_atom_shapes(self):
        self.assertEqual(blaze.dshape('bool'), blaze.bool_)
        self.assertEqual(blaze.dshape('int8'), blaze.i1)
        self.assertEqual(blaze.dshape('int16'), blaze.i2)
        self.assertEqual(blaze.dshape('int32'), blaze.i4)
        self.assertEqual(blaze.dshape('int64'), blaze.i8)
        self.assertEqual(blaze.dshape('uint8'), blaze.u1)
        self.assertEqual(blaze.dshape('uint16'), blaze.u2)
        self.assertEqual(blaze.dshape('uint32'), blaze.u4)
        self.assertEqual(blaze.dshape('uint64'), blaze.u8)
        self.assertEqual(blaze.dshape('float32'), blaze.f4)
        self.assertEqual(blaze.dshape('float64'), blaze.f8)
        self.assertEqual(blaze.dshape('complex64'), blaze.c8)
        self.assertEqual(blaze.dshape('complex128'), blaze.c16)

    def test_atom_shape_errors(self):
        self.assertRaises(TypeError, blaze.dshape, 'boot')
        self.assertRaises(TypeError, blaze.dshape, 'int33')
        self.assertRaises(TypeError, blaze.dshape, '12')

    def test_type_decl(self):
        self.assertRaises(TypeError, blaze.dshape, 'type X T = 3, T')
        self.assertEqual(blaze.dshape('3, int32'), blaze.dshape('type X = 3, int32'))

    def test_string_atom(self):
        self.assertEqual(blaze.dshape('string'), blaze.dshape("string('U8')"))
        self.assertEqual(blaze.dshape("string('ascii')").encoding, 'A')
        self.assertEqual(blaze.dshape("string('A')").encoding, 'A')
        self.assertEqual(blaze.dshape("string('utf-8')").encoding, 'U8')
        self.assertEqual(blaze.dshape("string('U8')").encoding, 'U8')
        self.assertEqual(blaze.dshape("string('utf-16')").encoding, 'U16')
        self.assertEqual(blaze.dshape("string('U16')").encoding, 'U16')
        self.assertEqual(blaze.dshape("string('utf-32')").encoding, 'U32')
        self.assertEqual(blaze.dshape("string('U32')").encoding, 'U32')

    def test_struct_of_array(self):
        self.assertEqual(str(blaze.dshape('5, int32')), '5, int32')
        self.assertEqual(str(blaze.dshape('{field: 5, int32}')), '{ field : 5, int32 }')
        self.assertEqual(str(blaze.dshape('{field: M, int32}')), '{ field : M, int32 }')

if __name__ == '__main__':
    unittest.main()
