import blaze
from blaze import datashape
import numpy as np
import unittest

class TestDatashapeCreation(unittest.TestCase):

    def test_raise_on_bad_input(self):
        # Make sure it raises exceptions on a few nonsense inputs
        self.assertRaises(TypeError, blaze.dshape, None)
        self.assertRaises(TypeError, blaze.dshape, lambda x: x+1)
        # Check issue 11
        self.assertRaises(datashape.parser.DatashapeSyntaxError, blaze.dshape, '1,')

    def test_reserved_future_int(self):
        # The "int" datashape is reserved for a future big integer type
        self.assertRaises(Exception, blaze.dshape, "int")

    def test_atom_shapes(self):
        self.assertEqual(blaze.dshape('bool'), datashape.bool_)
        self.assertEqual(blaze.dshape('int8'), datashape.int8)
        self.assertEqual(blaze.dshape('int16'), datashape.int16)
        self.assertEqual(blaze.dshape('int32'), datashape.int32)
        self.assertEqual(blaze.dshape('int64'), datashape.int64)
        self.assertEqual(blaze.dshape('uint8'), datashape.uint8)
        self.assertEqual(blaze.dshape('uint16'), datashape.uint16)
        self.assertEqual(blaze.dshape('uint32'), datashape.uint32)
        self.assertEqual(blaze.dshape('uint64'), datashape.uint64)
        self.assertEqual(blaze.dshape('float32'), datashape.float32)
        self.assertEqual(blaze.dshape('float64'), datashape.float64)
        self.assertEqual(blaze.dshape('complex64'), datashape.complex64)
        self.assertEqual(blaze.dshape('complex128'), datashape.complex128)
        self.assertEqual(blaze.dshape("string"), blaze.datashape.string)
        self.assertEqual(blaze.dshape("json"), blaze.datashape.json)

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
        self.assertEqual(str(blaze.dshape('{field: 5, int32}')),
                        '{ field : 5, int32 }')
        self.assertEqual(str(blaze.dshape('{field: M, int32}')),
                        '{ field : M, int32 }')

    def test_ragged_array(self):
        self.assertTrue(isinstance(blaze.dshape('3, var, int32')[1], datashape.Var))

if __name__ == '__main__':
    unittest.main()
