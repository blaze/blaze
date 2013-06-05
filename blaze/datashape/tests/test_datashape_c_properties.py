import blaze
from blaze import datashape
import numpy as np
import unittest
import ctypes

class TestDatashapeCreation(unittest.TestCase):
    def test_ctypes_sizeof(self):
        self.assertEqual(datashape.int8.c_itemsize, 1)
        self.assertEqual(datashape.int16.c_itemsize, 2)
        self.assertEqual(datashape.int32.c_itemsize, 4)
        self.assertEqual(datashape.int64.c_itemsize, 8)
        self.assertEqual(datashape.uint8.c_itemsize, 1)
        self.assertEqual(datashape.uint16.c_itemsize, 2)
        self.assertEqual(datashape.uint32.c_itemsize, 4)
        self.assertEqual(datashape.uint64.c_itemsize, 8)
        self.assertEqual(datashape.float32.c_itemsize, 4)
        self.assertEqual(datashape.float64.c_itemsize, 8)
        self.assertEqual(datashape.c_short.c_itemsize,
                        ctypes.sizeof(ctypes.c_short))
        self.assertEqual(datashape.c_int.c_itemsize,
                        ctypes.sizeof(ctypes.c_int))
        self.assertEqual(datashape.c_long.c_itemsize,
                        ctypes.sizeof(ctypes.c_long))
        self.assertEqual(datashape.c_longlong.c_itemsize,
                        ctypes.sizeof(ctypes.c_longlong))
        self.assertEqual(datashape.c_intptr.c_itemsize,
                        ctypes.sizeof(ctypes.c_void_p))
        self.assertEqual(datashape.c_uintptr.c_itemsize,
                        ctypes.sizeof(ctypes.c_void_p))

    def test_ctypes_alignment(self):
        self.assertEqual(datashape.int8.c_alignment,
                        ctypes.alignment(ctypes.c_int8))
        self.assertEqual(datashape.int16.c_alignment,
                        ctypes.alignment(ctypes.c_int16))
        self.assertEqual(datashape.int32.c_alignment,
                        ctypes.alignment(ctypes.c_int32))
        self.assertEqual(datashape.int64.c_alignment,
                        ctypes.alignment(ctypes.c_int64))
        self.assertEqual(datashape.uint8.c_alignment,
                        ctypes.alignment(ctypes.c_uint8))
        self.assertEqual(datashape.uint16.c_alignment,
                        ctypes.alignment(ctypes.c_uint16))
        self.assertEqual(datashape.uint32.c_alignment,
                        ctypes.alignment(ctypes.c_uint32))
        self.assertEqual(datashape.uint64.c_alignment,
                        ctypes.alignment(ctypes.c_uint64))
        self.assertEqual(datashape.float32.c_alignment,
                        ctypes.alignment(ctypes.c_float))
        self.assertEqual(datashape.float64.c_alignment,
                        ctypes.alignment(ctypes.c_double))

    def test_string(self):
        class StringData(ctypes.Structure):
            _fields_ = [('begin', ctypes.c_void_p),
                        ('end', ctypes.c_void_p)]
        # string
        self.assertEqual(datashape.string.c_itemsize,
                        ctypes.sizeof(StringData))
        self.assertEqual(datashape.string.c_alignment,
                        ctypes.alignment(StringData))
        # bytes
        self.assertEqual(datashape.bytes_.c_itemsize,
                        ctypes.sizeof(StringData))
        self.assertEqual(datashape.bytes_.c_alignment,
                        ctypes.alignment(StringData))
        # json
        self.assertEqual(datashape.json.c_itemsize,
                        ctypes.sizeof(StringData))
        self.assertEqual(datashape.json.c_alignment,
                        ctypes.alignment(StringData))

    def test_array(self):
        ds = datashape.dshape('3, 5, 2, int32')
        self.assertEqual(ds.c_itemsize, 3 * 5 * 2 * 4)
        self.assertEqual(ds.c_alignment, datashape.int32.c_alignment)
        self.assertEqual(ds.c_strides, (40, 8, 4))

    def test_array_nolayout(self):
        # If the datashape has no layout, it should raise errors
        ds = datashape.dshape('3, 5, M, int32')
        self.assertRaises(AttributeError, lambda x : x.c_itemsize, ds)
        self.assertRaises(AttributeError, lambda x : x.c_alignment, ds)
        self.assertRaises(AttributeError, lambda x : x.c_strides, ds)

    def test_record(self):
        class ctds(ctypes.Structure):
            _fields_ = [('a', ctypes.c_int8),
                        ('b', ctypes.c_double),
                        ('c', ctypes.c_uint8),
                        ('d', ctypes.c_uint16)]
        ds = datashape.dshape('{a: int8; b: float64; c: uint8; d: float16}')
        self.assertEqual(ds.c_itemsize, ctypes.sizeof(ctds))
        self.assertEqual(ds.c_alignment, ctypes.alignment(ctds))
        self.assertEqual(ds.c_offsets,
                        (ctds.a.offset, ctds.b.offset, ctds.c.offset, ctds.d.offset))

    def test_record_nolayout(self):
        ds = datashape.dshape('{a: int8; b: M, float32}')
        self.assertRaises(AttributeError, lambda x : x.c_itemsize, ds)
        self.assertRaises(AttributeError, lambda x : x.c_alignment, ds)
        self.assertRaises(AttributeError, lambda x : x.c_offsets, ds)

if __name__ == '__main__':
    unittest.main()
