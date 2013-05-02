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
        self.assertIsInstance(npdd, DataDescriptor)

    def test_descriptor_iter_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Iteration should produce NumPyDataDescriptor instances
        for el in npdd:
            self.assertIsInstance(el, NumPyDataDescriptor)
            self.assertIsInstance(el, DataDescriptor)

    def test_descriptor_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Indexing should produce NumPyDataDescriptor instances
        self.assertIsInstance(npdd[0], NumPyDataDescriptor)
        self.assertIsInstance(npdd[1,2], NumPyDataDescriptor)

    def test_element_iter_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Requesting element iteration should produce an
        # IElementIter object
        ei = npdd.element_iter_interface()
        self.assertIsInstance(ei, IElementIter)
        # Iteration over the IElementIter object should produce
        # ctypes c_void_p pointers
        for ptr in ei:
            self.assertIsInstance(ptr, ctypes.c_void_p)

    def test_element_getitem_types(self):
        a = np.arange(6).reshape(2,3)
        npdd = NumPyDataDescriptor(a)

        # Requesting get_element with one index should produce an
        # IGetElement object
        ge = npdd.get_element_interface(1)
        self.assertIsInstance(ge, IGetElement)
        # Indexing the IGetElement object should produce
        # ctypes c_void_p pointers
        self.assertIsInstance(ge.get((1,)), ctypes.c_void_p)

        # Requesting get_element with two indices should produce an
        # IGetElement object
        ge = npdd.get_element_interface(2)
        self.assertIsInstance(ge, IGetElement)
        # Indexing the IGetElement object should produce
        # ctypes c_void_p pointers
        self.assertIsInstance(ge.get((1,2)), ctypes.c_void_p)

if __name__ == '__main__':
    unittest.main()
