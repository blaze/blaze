from blaze.datadescriptor import HDF5_DDesc, DyND_DDesc, DDesc
from blaze.datadescriptor.util import openfile
import unittest
import tempfile
import os
from dynd import nd
import h5py
import numpy as np
from sys import stdout


class SingleTestClass(unittest.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp('h5')

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    """
    def test_creation(self):
        dd = HDF5_DDesc(self.filename, 'data', 'w', dshape='2 * 2 * int32')

        with h5py.File(self.filename, 'r') as f:
            d = f['data']
            self.assertEquals(d.dtype.name, 'int32')

        self.assertRaises(ValueError, lambda: HDF5_DDesc('bar.hdf5', 'foo'))
        """

    def test_existing_array(self):
        print("existing array")
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5_DDesc(self.filename, '/data', mode='a')

        known = {'chunks': True,
                 'maxshape': (None, 3),
                 'compression': None}
        attrs = dd.attributes()
        assert attrs['chunks']
        self.assertEquals(attrs['maxshape'], (None, 3))
        assert not attrs['compression']

        self.assertEquals(str(dd.dshape), 'var * 3 * int32')

    def test_extend_chunks(self):
        print("extend chunks")
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5_DDesc(self.filename, '/data', mode='a')

        chunks = [nd.array([[1, 2, 3]], dtype='1 * 3 * int32'),
                  nd.array([[4, 5, 6]], dtype='1 * 3 * int32')]

        dd.extend_chunks(chunks)

        result = dd.dynd_arr()[-2:, :]
        expected = nd.array([[1, 2, 3],
                             [4, 5, 6]], dtype='strided * strided * int32')

        self.assertEquals(nd.as_py(result), nd.as_py(expected))

    def test_iterchunks(self):
        print("iterchunks")
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i8')
            d[:] = 1
        dd = HDF5_DDesc(self.filename, '/data')
        assert all(isinstance(chunk, nd.array) for chunk in dd.iterchunks())
