import unittest

import blaze
from blaze import datashape
from blaze import dshape

class TestDataShapeStr(unittest.TestCase):
    def test_primitive_measure_str(self):
        self.assertEqual(str(datashape.int8), 'int8')
        self.assertEqual(str(datashape.int16), 'int16')
        self.assertEqual(str(datashape.int32), 'int32')
        self.assertEqual(str(datashape.int64), 'int64')
        self.assertEqual(str(datashape.uint8), 'uint8')
        self.assertEqual(str(datashape.uint16), 'uint16')
        self.assertEqual(str(datashape.uint32), 'uint32')
        self.assertEqual(str(datashape.uint64), 'uint64')
        self.assertEqual(str(datashape.float32), 'float32')
        self.assertEqual(str(datashape.float64), 'float64')
        self.assertEqual(str(datashape.string), 'string')
        self.assertEqual(str(datashape.String(3)), 'string(3)')
        self.assertEqual(str(datashape.String('A')), "string('A')")

    def test_structure_str(self):
        self.assertEqual(str(dshape('{x:int32; y:int64}')),
                        '{ x : int32; y : int64 }')

    def test_array_str(self):
        self.assertEqual(str(dshape('3,5,int16')),
                        '3, 5, int16')

    def test_primitive_measure_repr(self):
        self.assertEqual(repr(datashape.int8), 'dshape("int8")')
        self.assertEqual(repr(datashape.int16), 'dshape("int16")')
        self.assertEqual(repr(datashape.int32), 'dshape("int32")')
        self.assertEqual(repr(datashape.int64), 'dshape("int64")')
        self.assertEqual(repr(datashape.uint8), 'dshape("uint8")')
        self.assertEqual(repr(datashape.uint16), 'dshape("uint16")')
        self.assertEqual(repr(datashape.uint32), 'dshape("uint32")')
        self.assertEqual(repr(datashape.uint64), 'dshape("uint64")')
        self.assertEqual(repr(datashape.float32), 'dshape("float32")')
        self.assertEqual(repr(datashape.float64), 'dshape("float64")')
        self.assertEqual(repr(datashape.string), 'dshape("string")')
        self.assertEqual(repr(datashape.String(3)), 'dshape("string(3)")')
        self.assertEqual(repr(datashape.String('A')),
                        'dshape("string(\'A\')")')

    def test_structure_repr(self):
        self.assertEqual(repr(dshape('{x:int32; y:int64}')),
                        'dshape("{ x : int32; y : int64 }")')

    def test_array_repr(self):
        self.assertEqual(repr(dshape('3,5,int16')),
                        'dshape("3, 5, int16")')

if __name__ == '__main__':
    unittest.main()

