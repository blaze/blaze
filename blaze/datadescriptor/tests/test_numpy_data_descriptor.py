import unittest

import numpy as np
import blaze
from blaze import datashape
from blaze.datadescriptor import (NumPyDataDescriptor,
                IDataDescriptor, IElementReader, IElementReadIter,
                dd_as_py)
from ...py3help import _inttypes
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

if __name__ == '__main__':
    unittest.main()
