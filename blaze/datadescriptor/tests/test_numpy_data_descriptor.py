import unittest

import numpy as np
import blaze
from blaze import datashape
from blaze.datadescriptor import (NumPyDataDescriptor,
                IDataDescriptor, IElementReader, IElementReadIter,
                IElementWriter, IElementWriteIter,
                dd_as_py)
from blaze.py3help import _inttypes, izip
import ctypes

class TestNumPyDataDescriptor(unittest.TestCase):
    def test_basic_object_type(self):
        self.assertTrue(issubclass(NumPyDataDescriptor, IDataDescriptor))
        a = np.arange(6).reshape(2,3)
        dd = NumPyDataDescriptor(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [[0,1,2], [3,4,5]])

    def test_descriptor_iter_types(self):
        a = np.arange(6).reshape(2,3)
        dd = NumPyDataDescriptor(a)

        # Iteration should produce NumPyDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, NumPyDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [[0,1,2], [3,4,5]])

    def test_descriptor_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        dd = NumPyDataDescriptor(a)

        # Indexing should produce NumPyDataDescriptor instances
        self.assertTrue(isinstance(dd[0], NumPyDataDescriptor))
        self.assertEqual(dd_as_py(dd[0]), [0,1,2])
        self.assertTrue(isinstance(dd[1,2], NumPyDataDescriptor))
        self.assertEqual(dd_as_py(dd[1,2]), 5)

    def test_element_iter_types(self):
        a = np.arange(6).reshape(2,3)
        dd = NumPyDataDescriptor(a)

        # Requesting element iteration should produce an
        # IElementReadIter object
        ei = dd.element_read_iter()
        self.assertTrue(isinstance(ei, IElementReadIter))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        for ptr in ei:
            self.assertTrue(isinstance(ptr, _inttypes))

    def test_element_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        dd = NumPyDataDescriptor(a)

        # Requesting get_element with one index should produce an
        # IElementReader object
        ge = dd.element_reader(1)
        self.assertTrue(isinstance(ge, IElementReader))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        self.assertTrue(isinstance(ge.read_single((1,)), _inttypes))

        # Requesting element reader with two indices should produce an
        # IElementReader object
        ge = dd.element_reader(2)
        self.assertTrue(isinstance(ge, IElementReader))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        self.assertTrue(isinstance(ge.read_single((1,2)), _inttypes))

    def test_element_write(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        dd = NumPyDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int32'))
        ge = dd.element_writer(1)
        self.assertTrue(isinstance(ge, IElementWriter))

        x = ctypes.c_int32(123)
        ge.write_single((1,), ctypes.addressof(x))
        self.assertEqual(dd_as_py(dd), [1,123,3,4,5])

        with ge.buffered_ptr((3,)) as dst_ptr:
            x = ctypes.c_int32(456)
            ctypes.memmove(dst_ptr, ctypes.addressof(x), 4)
        self.assertEqual(dd_as_py(dd), [1,123,3,456,5])

    def test_element_iter_write(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        dd = NumPyDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int32'))
        with dd.element_write_iter() as ge:
            self.assertTrue(isinstance(ge, IElementWriteIter))
            for val, ptr in izip([5,7,4,5,3], ge):
                x = ctypes.c_int32(val)
                ctypes.memmove(ptr, ctypes.addressof(x), 4)
        self.assertEqual(dd_as_py(dd), [5,7,4,5,3])

    def test_element_write_buffered(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.dtype(np.int32).newbyteorder())
        dd = NumPyDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int32'))
        self.assertFalse(dd.npyarr.dtype.isnative)
        ge = dd.element_writer(1)
        self.assertTrue(isinstance(ge, IElementWriter))

        x = ctypes.c_int32(123)
        ge.write_single((1,), ctypes.addressof(x))
        self.assertEqual(dd_as_py(dd), [1,123,3,4,5])

        with ge.buffered_ptr((3,)) as dst_ptr:
            x = ctypes.c_int32(456)
            ctypes.memmove(dst_ptr, ctypes.addressof(x), 4)
        self.assertEqual(dd_as_py(dd), [1,123,3,456,5])

    def test_element_iter_write_buffered(self):
        a = np.array([1, 2, 3, 4, 5], dtype=np.dtype(np.int32).newbyteorder())
        dd = NumPyDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int32'))
        with dd.element_write_iter() as ge:
            self.assertTrue(isinstance(ge, IElementWriteIter))
            for val, ptr in izip([5,7,4,5,3], ge):
                x = ctypes.c_int64(val)
                ctypes.memmove(ptr, ctypes.addressof(x), 8)
        self.assertEqual(dd_as_py(dd), [5,7,4,5,3])

if __name__ == '__main__':
    unittest.main()
