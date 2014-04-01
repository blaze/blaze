from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os

import datashape
import numpy as np

from blaze.datadescriptor import (
    HDF5_DDesc, DyND_DDesc, DDesc, ddesc_as_py)
from blaze.py2help import skipIf

from blaze.optional_packages import tables_is_here
if tables_is_here:
    import tables as tb


class TestHDF5DDesc(unittest.TestCase):

    def setUp(self):
        handle, self.hdf5_file = tempfile.mkstemp(".h5")
        os.close(handle)  # close the non needed file handle
        self.a1 = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        self.a2 = np.array([[1, 2, 3], [3, 2, 1]], dtype="int64")
        self.t1 = np.array([(1, 2, 3), (3, 2, 1)], dtype="i4,i8,f8")
        with tb.open_file(self.hdf5_file, "w") as f:
            f.create_array(f.root, 'a1', self.a1)
            f.create_table(f.root, 't1', self.t1)
            f.create_group(f.root, 'g')
            f.create_array(f.root.g, 'a2', self.a2)

    def tearDown(self):
        os.remove(self.hdf5_file)

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_basic_object_type(self):
        self.assertTrue(issubclass(HDF5_DDesc, DDesc))
        dd = HDF5_DDesc(self.hdf5_file, '/a1')
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, DDesc))
        self.assertEqual(ddesc_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_iter_types(self):
        dd = HDF5_DDesc(self.hdf5_file, '/a1')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int32'))
        # Iteration should produce DyND_DDesc instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.append(ddesc_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_getitem_types(self):
        dd = HDF5_DDesc(self.hdf5_file, '/g/a2')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int64'))
        # Indexing should produce DyND_DDesc instances
        self.assertTrue(isinstance(dd[0], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[1,2]), 1)

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_setitem(self):
        dd = HDF5_DDesc(self.hdf5_file, '/g/a2', mode='a')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int64'))
        dd[1,2] = 10
        self.assertEqual(ddesc_as_py(dd[1,2]), 10)
        dd[1] = [10, 11, 12]
        self.assertEqual(ddesc_as_py(dd[1]), [10, 11, 12])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_append(self):
        dd = HDF5_DDesc(self.hdf5_file, '/t1', mode='a')

        tshape = datashape.dshape(
            '2 * { f0 : int32, f1 : int64, f2 : float64 }')
        self.assertEqual(dd.dshape, tshape)
        dd.append([(10, 11, 12)])
        dvals = {'f0': 10, 'f1': 11, 'f2': 12.}
        rvals = ddesc_as_py(dd[2])
        is_equal = [(rvals[k] == dvals[k]) for k in dvals]
        self.assertEqual(is_equal, [True]*3)


if __name__ == '__main__':
    unittest.main()
