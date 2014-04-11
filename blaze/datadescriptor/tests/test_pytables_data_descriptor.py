from __future__ import absolute_import, division, print_function

import unittest
import tempfile
import os

import datashape
import numpy as np
from dynd import nd

from blaze.datadescriptor import (
    PyTables_DDesc, DyND_DDesc, DDesc, ddesc_as_py)
from blaze.py2help import skipIf
from blaze.datadescriptor.util import openfile
import h5py

from blaze.optional_packages import tables_is_here
if tables_is_here:
    import tables as tb


class TestPyTablesDDesc(unittest.TestCase):

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
        self.assertTrue(issubclass(PyTables_DDesc, DDesc))
        dd = PyTables_DDesc(self.hdf5_file, '/a1')
        # Make sure the right type is returned
        self.assertTrue(isinstance(dd, DDesc))
        self.assertEqual(ddesc_as_py(dd), [[1, 2, 3], [4, 5, 6]])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_iter_types(self):
        dd = PyTables_DDesc(self.hdf5_file, '/a1')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int32'))
        # Iteration should produce DyND_DDesc instances
        vals = []
        for el in dd:
            self.assertTrue(isinstance(el, DyND_DDesc))
            self.assertTrue(isinstance(el, DDesc))
            vals.append(ddesc_as_py(el))
        self.assertEqual(vals, [[1, 2, 3], [4, 5, 6]])

    def test_iterchunks(self):
        dd = PyTables_DDesc(self.hdf5_file, '/a1')
        self.assertEqual(len(list(dd.iterchunks(blen=1))), 2)
        assert all(isinstance(chunk, DDesc) for chunk in dd.iterchunks())

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_getitem_types(self):
        dd = PyTables_DDesc(self.hdf5_file, '/g/a2')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int64'))
        # Indexing should produce DyND_DDesc instances
        self.assertTrue(isinstance(dd[0], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[0]), [1,2,3])
        self.assertTrue(isinstance(dd[1,2], DyND_DDesc))
        self.assertEqual(ddesc_as_py(dd[1,2]), 1)

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_setitem(self):
        dd = PyTables_DDesc(self.hdf5_file, '/g/a2', mode='a')

        self.assertEqual(dd.dshape, datashape.dshape('2 * 3 * int64'))
        dd[1,2] = 10
        self.assertEqual(ddesc_as_py(dd[1,2]), 10)
        dd[1] = [10, 11, 12]
        self.assertEqual(ddesc_as_py(dd[1]), [10, 11, 12])

    @skipIf(not tables_is_here, 'pytables is not installed')
    def test_descriptor_append(self):
        dd = PyTables_DDesc(self.hdf5_file, '/t1', mode='a')

        tshape = datashape.dshape(
            '2 * { f0 : int32, f1 : int64, f2 : float64 }')
        self.assertEqual(dd.dshape, tshape)
        dd.append([(10, 11, 12)])
        dvals = {'f0': 10, 'f1': 11, 'f2': 12.}
        rvals = ddesc_as_py(dd[2])
        is_equal = [(rvals[k] == dvals[k]) for k in dvals]
        self.assertEqual(is_equal, [True]*3)


def test_extend_chunks():
    with openfile() as path:
        with h5py.File(path, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = PyTables_DDesc(path, '/data', mode='a')

        chunks = [DyND_DDesc(nd.array([[1, 2, 3]], dtype='1 * 3 * int32')),
                  DyND_DDesc(nd.array([[4, 5, 6]], dtype='1 * 3 * int32'))]

        dd.extend_chunks(chunks)

        result = dd.dynd_arr()[-2:, :]
        expected = nd.array([[1, 2, 3],
                             [4, 5, 6]], dtype='strided * strided * int32')

        print(repr(result))
        print(repr(expected))

        assert nd.as_py(result) == nd.as_py(expected)

def test_iterchunks():
    with openfile() as path:
        with h5py.File(path, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i8')
            d[:] = 1
        dd = PyTables_DDesc(path, '/data')
        assert all(isinstance(chunk, DDesc) for chunk in dd.iterchunks())


if __name__ == '__main__':
    unittest.main()
