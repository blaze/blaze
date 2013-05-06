import unittest

import numpy as np
import blaze
from blaze import datashape
from blaze.datadescriptor import NumPyDataDescriptor, \
                DataDescriptor, IGetElement, IElementIter
import ctypes

class TestNumPyDataDescriptor(unittest.TestCase):
    def test_basic_object_type(self):
        self.assertTrue(issubclass(NumPyDataDescriptor, DataDescriptor))
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)
        # Make sure the right type is returned
        self.assertTrue(isinstance(npdd, DataDescriptor))

    def test_descriptor_iter_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Iteration should produce NumPyDataDescriptor instances
        for el in npdd:
            self.assertTrue(isinstance(el, NumPyDataDescriptor))
            self.assertTrue(isinstance(el, DataDescriptor))

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
        # IElementIter object
        ei = npdd.element_iter_interface()
        self.assertTrue(isinstance(ei, IElementIter))
        # Iteration over the IElementIter object should produce
        # ctypes c_void_p pointers
        for ptr in ei:
            self.assertTrue(isinstance(ptr, ctypes.c_void_p))

    def test_element_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Requesting get_element with one index should produce an
        # IGetElement object
        ge = npdd.get_element_interface(1)
        self.assertTrue(isinstance(ge, IGetElement))
        # Indexing the IGetElement object should produce
        # ctypes c_void_p pointers
        self.assertTrue(isinstance(ge.get((1,)), ctypes.c_void_p))

        # Requesting get_element with two indices should produce an
        # IGetElement object
        ge = npdd.get_element_interface(2)
        self.assertTrue(isinstance(ge, IGetElement))
        # Indexing the IGetElement object should produce
        # ctypes c_void_p pointers
        self.assertTrue(isinstance(ge.get((1,2)), ctypes.c_void_p))

if __name__ == '__main__':
    unittest.main()
