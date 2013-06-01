import unittest
import sys
import blaze
from blaze import datashape
from blaze.datadescriptor import (DyNDDataDescriptor,
                IDataDescriptor, IElementReader, IElementReadIter,
                IElementWriter, IElementWriteIter,
                dd_as_py)
from blaze.py3help import _inttypes, skipIf, izip
import ctypes

try:
    import dynd
    from dynd import nd, ndt
except ImportError:
    dynd = None

class TestDyNDDataDescriptor(unittest.TestCase):
    @skipIf(dynd is None, 'dynd is not installed')
    def test_basic_object_type(self):
        self.assertTrue(issubclass(DyNDDataDescriptor, IDataDescriptor))
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_descriptor_iter_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_descriptor_getitem_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Indexing should produce DyNDDataDescriptor instances
        self.assertTrue(isinstance(dd[0], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[1,2]), 6)

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_iter_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Requesting element iteration should produce an
        # IElementReadIter object
        ei = dd.element_read_iter()
        self.assertTrue(isinstance(ei, IElementReadIter))
        # Iteration over the IElementReadIter object should produce
        # raw ints which are pointers
        for ptr in ei:
            self.assertTrue(isinstance(ptr, _inttypes))

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_getitem_types(self):
        a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Requesting element_reader with one index should produce an
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

    @skipIf(dynd is None, 'dynd is not installed')
    def test_var_dim(self):
        a = nd.ndobject([[1,2,3], [4,5], [6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('3, var, int32'))
        self.assertEqual(dd_as_py(dd), [[1,2,3], [4,5], [6]])
        self.assertEqual(dd_as_py(dd[0]), [1,2,3])
        self.assertEqual(dd_as_py(dd[1]), [4,5])
        self.assertEqual(dd_as_py(dd[2]), [6])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_write(self):
        a = nd.ndobject([1, 2, 3, 4, 5])
        dd = DyNDDataDescriptor(a)

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

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_iter_write(self):
        a = nd.ndobject([1, 2, 3, 4, 5])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int32'))
        with dd.element_write_iter() as ge:
            self.assertTrue(isinstance(ge, IElementWriteIter))
            for val, ptr in izip([5,7,4,5,3], ge):
                x = ctypes.c_int32(val)
                ctypes.memmove(ptr, ctypes.addressof(x), 4)
        self.assertEqual(dd_as_py(dd), [5,7,4,5,3])


    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_write_buffered(self):
        a = nd.ndobject([1, 2, 3, 4, 5]).ucast(ndt.int64)
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int64'))
        ge = dd.element_writer(1)
        self.assertTrue(isinstance(ge, IElementWriter))

        x = ctypes.c_int32(123)
        ge.write_single((1,), ctypes.addressof(x))
        self.assertEqual(dd_as_py(dd), [1,123,3,4,5])

        with ge.buffered_ptr((3,)) as dst_ptr:
            x = ctypes.c_int32(456)
            ctypes.memmove(dst_ptr, ctypes.addressof(x), 4)
        self.assertEqual(dd_as_py(dd), [1,123,3,456,5])

    @skipIf(dynd is None, 'dynd is not installed')
    def test_element_iter_write_buffered(self):
        a = nd.ndobject([1, 2, 3, 4, 5]).ucast(ndt.int64)
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('5, int64'))
        with dd.element_write_iter() as ge:
            self.assertTrue(isinstance(ge, IElementWriteIter))
            for val, ptr in izip([5,7,4,5,3], ge):
                x = ctypes.c_int64(val)
                ctypes.memmove(ptr, ctypes.addressof(x), 8)
        self.assertEqual(dd_as_py(dd), [5,7,4,5,3])

if __name__ == '__main__':
    unittest.main()
