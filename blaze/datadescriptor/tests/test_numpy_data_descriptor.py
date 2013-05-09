import unittest

import numpy as np
import blaze
from blaze import datashape
from blaze.datadescriptor import NumPyDataDescriptor, \
                IDataDescriptor, IElementReader, IElementReadIter
import ctypes

class TestNumPyDataDescriptor(unittest.TestCase):
    def test_basic_object_type(self):
        self.assertTrue(issubclass(NumPyDataDescriptor, IDataDescriptor))
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(npdd, IDataDescriptor))

    def test_descriptor_iter_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Iteration should produce NumPyDataDescriptor instances
        for el in npdd:
            self.assertTrue(isinstance(el, NumPyDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))

    def test_descriptor_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Indexing should produce NumPyDataDescriptor instances
        self.assertTrue(isinstance(npdd[0], NumPyDataDescriptor))
        self.assertTrue(isinstance(npdd[1,2], NumPyDataDescriptor))

    def test_element_iter_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Requesting element iteration should produce an
        # IElementReadIter object
        ei = npdd.element_read_iter()
        self.assertTrue(isinstance(ei, IElementReadIter))
        # Iteration over the IElementReadIter object should produce
        # ctypes c_void_p pointers
        for ptr in ei:
            self.assertTrue(isinstance(ptr, ctypes.c_void_p))

    def test_element_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Requesting get_element with one index should produce an
        # IElementReader object
        ge = npdd.element_reader(1)
        self.assertTrue(isinstance(ge, IElementReader))
        # Indexing the IElementReader object should produce
        # ctypes c_void_p pointers
        self.assertTrue(isinstance(ge.read_single((1,)), ctypes.c_void_p))

        # Requesting element reader with two indices should produce an
        # IElementReader object
        ge = npdd.element_reader(2)
        self.assertTrue(isinstance(ge, IElementReader))
        # Indexing the IElementReader object should produce
        # ctypes c_void_p pointers
        self.assertTrue(isinstance(ge.read_single((1,2)), ctypes.c_void_p))

if __name__ == '__main__':
    unittest.main()
