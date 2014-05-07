import unittest
import tempfile
import os
from dynd import nd
import h5py
import numpy as np
from sys import stdout
from blaze.compatibility import skip

from blaze.data import HDF5
from blaze.utils import tmpfile

class SingleTestClass(unittest.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp('h5')

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    @skip("This runs fine in isolation, segfaults in full test")
    def test_creation(self):
        dd = HDF5(self.filename, 'data', 'w', dshape='2 * 2 * int32')

        with h5py.File(self.filename, 'r') as f:
            d = f['data']
            self.assertEquals(d.dtype.name, 'int32')

        self.assertRaises(ValueError, lambda: HDF5('bar.hdf5', 'foo'))

    def test_existing_array(self):
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5(self.filename, '/data', mode='a')

        known = {'chunks': True,
                 'maxshape': (None, 3),
                 'compression': None}
        attrs = dd.attributes()
        assert attrs['chunks']
        self.assertEquals(attrs['maxshape'], (None, 3))
        assert not attrs['compression']

        self.assertEquals(str(dd.dshape), 'var * 3 * int32')

        print(dd.as_py())
        self.assertEqual(dd.as_py(), ((1, 1, 1), (1, 1, 1), (1, 1, 1)))

    def test_extend_chunks(self):
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5(self.filename, '/data', mode='a')

        chunks = [nd.array([[1, 2, 3]], dtype='1 * 3 * int32'),
                  nd.array([[4, 5, 6]], dtype='1 * 3 * int32')]

        dd.extend_chunks(chunks)

        result = dd.as_dynd()[-2:, :]
        expected = nd.array([[1, 2, 3],
                             [4, 5, 6]], dtype='strided * strided * int32')

        self.assertEquals(nd.as_py(result), nd.as_py(expected))

    def test_chunks(self):
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i8')
            d[:] = 1
        dd = HDF5(self.filename, '/data')
        assert all(isinstance(chunk, nd.array) for chunk in dd.chunks())

    @skip("This runs fine in isolation, segfaults in full test")
    def test_extend(self):
        dd = HDF5(self.filename, '/data', 'a', schema='2 * int32')
        dd.extend([(1, 1), (2, 2)])

        results = list(dd)

        self.assertEquals(nd.as_py(results[0]), (1, 1))
        self.assertEquals(nd.as_py(results[1]), (2, 2))

    @skip("This runs fine in isolation, segfaults in full test")
    def test_schema(self):
        dd = HDF5(self.filename, '/data', 'a', schema='2 * int32')

        self.assertEquals(str(dd.schema), '2 * int32')
        self.assertEquals(str(dd.dshape), 'var * 2 * int32')

    @skip("This runs fine in isolation, segfaults in full test")
    def test_dshape(self):
        dd = HDF5(self.filename, '/data', 'a', dshape='var * 2 * int32')

        self.assertEquals(str(dd.schema), '2 * int32')
        self.assertEquals(str(dd.dshape), 'var * 2 * int32')

    @skip("This runs fine in isolation, segfaults in full test")
    def test_setitem(self):
        dd = HDF5(self.filename, 'data', 'a', dshape='2 * 2 * 2 * int')
        dd[:] = 1
        dd[0, 0, :] = 2
        self.assertEqual(nd.as_py(dd.as_dynd()), [[[2, 2], [1, 1]],
                                                  [[1, 1], [1, 1]]])
