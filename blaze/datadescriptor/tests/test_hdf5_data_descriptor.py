import unittest
import tempfile
import sys
import os
import io

import datashape
from dynd import nd
import numpy as np
from blaze.datadescriptor import HDF5DataDescriptor, IDataDescriptor, dd_as_py

import tables as tb

class TestHDF5DataDescriptor(unittest.TestCase):

    def setUp(self):
        handle, self.hdf5_file = tempfile.mkstemp(".h5")
        self.a1 = np.array([[1, 2, 3], [4, 5, 6]])
        self.a2 = np.array([[1, 2, 3], [3, 2, 1]])
        with tb.open_file(self.hdf5_file, "w") as f:
            f.create_array(f.root, 'a1', self.a1)
            f.create_group(f.root, 'g')
            f.create_array(f.root.g, 'a2', self.a2)

    def tearDown(self):
        os.remove(self.hdf5_file)

    def test_basic_object_type(self):
        self.assertTrue(issubclass(HDF5DataDescriptor, IDataDescriptor))
        dd = HDF5DataDescriptor(self.hdf5_file, '/a1')
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, IDataDescriptor))
        self.assertEqual(dd_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    def _test_descriptor_iter_types(self):
        a = nd.array([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Iteration should produce DyNDDataDescriptor instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyNDDataDescriptor))
            self.assertTrue(isinstance(el, IDataDescriptor))
            vals.append(dd_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    def _test_descriptor_getitem_types(self):
        a = nd.array([[1, 2, 3], [4, 5, 6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('2, 3, int32'))
        # Indexing should produce DyNDDataDescriptor instances
        self.assertTrue(isinstance(dd[0], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyNDDataDescriptor))
        self.assertEqual(dd_as_py(dd[1,2]), 6)

    def _test_var_dim(self):
        a = nd.array([[1, 2, 3], [4, 5], [6]])
        dd = DyNDDataDescriptor(a)

        self.assertEqual(dd.dshape, datashape.dshape('3, var, int32'))
        self.assertEqual(dd_as_py(dd), [[1, 2, 3], [4, 5], [6]])
        self.assertEqual(dd_as_py(dd[0]), [1, 2, 3])
        self.assertEqual(dd_as_py(dd[1]), [4, 5])
        self.assertEqual(dd_as_py(dd[2]), [6])


if __name__ == '__main__':
    unittest.main()
