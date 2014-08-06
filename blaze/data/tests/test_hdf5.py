import unittest
import tempfile
import os
from dynd import nd
import h5py
import numpy as np
from sys import stdout
from datetime import date, datetime
from datashape import dshape
import pytest

from blaze.api.into import into
from blaze.data.hdf5 import HDF5, discover
from blaze.compatibility import unicode, xfail


class MakeFile(unittest.TestCase):
    def setUp(self):
        self.filename = tempfile.mktemp('h5')

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)


class SingleTestClass(MakeFile):
    def test_creation(self):
        dd = HDF5(self.filename, 'data', dshape='2 * 2 * int32')

        with h5py.File(self.filename, 'r') as f:
            d = f['data']
            self.assertEquals(d.dtype.name, 'int32')

        self.assertRaises(Exception, lambda: HDF5('bar.hdf5', 'foo'))

    def test_existing_array(self):
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5(self.filename, '/data')

        known = {'chunks': True,
                 'maxshape': (None, 3),
                 'compression': None}
        attrs = dd.attributes()
        assert attrs['chunks']
        self.assertEquals(attrs['maxshape'], (None, 3))
        assert not attrs['compression']

        self.assertEquals(str(dd.dshape), 'var * 3 * int32')

        self.assertEqual(tuple(map(tuple, dd.as_py())),
                         ((1, 1, 1), (1, 1, 1), (1, 1, 1)))

    def test_strings(self):
        stdout.flush()
        dt = h5py.special_dtype(vlen=unicode)
        with h5py.File(self.filename, 'w') as f:
            dtype = [('a', 'i4'), ('b', dt)]
            d = f.create_dataset('data', (3,), dtype=dtype,
                                 chunks=True, maxshape=(None,))
            x = np.array([(1, 'Hello'), (2, 'world'), (3, '!')],
                        dtype=[('a', 'i4'), ('b', 'O')])
            d[:] = x

    def test_extend_strings(self):
        stdout.flush()
        dt = h5py.special_dtype(vlen=unicode)
        dd = HDF5(self.filename, '/data',
                  schema='{a: int32, b: string}')

        dd.extend([(1, 'Hello'), (2, 'World!')])

    def test_coercion_after_creation(self):
        stdout.flush()
        dd = HDF5(self.filename, '/data',
                  schema='{a: int32, b: int}')

        dd.extend([(1, 10), (2, 20)])

        dd2 = HDF5(self.filename, '/data',
                   schema='{a: real, b: string}')

        self.assertEqual(list(dd2),
                          [(1.0, '10'), (2.0, '20')])

        self.assertEqual(nd.as_py(next(dd2.chunks()), tuple=True),
                         [(1.0, '10'), (2.0, '20')])


    def test_extend_chunks(self):
        stdout.flush()
        with h5py.File(self.filename, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i4',
                                 chunks=True, maxshape=(None, 3))
            d[:] = 1

        dd = HDF5(self.filename, '/data')

        chunks = [nd.array([[1, 2, 3]], dtype='1 * 3 * int32'),
                  nd.array([[4, 5, 6]], dtype='1 * 3 * int32')]

        dd.extend_chunks(chunks)

        result = dd.as_dynd()[-2:, :]
        expected = nd.array([[1, 2, 3],
                             [4, 5, 6]], dtype='strided * strided * int32')

        self.assertEquals(nd.as_py(result), nd.as_py(expected))

    def test_chunks(self):
        stdout.flush()
        with h5py.File(self.filename) as f:
            d = f.create_dataset('data', (3, 3), dtype='i8')
            d[:] = 1
        dd = HDF5(self.filename, '/data')
        assert all(isinstance(chunk, nd.array) for chunk in dd.chunks())

    def test_extend(self):
        dd = HDF5(self.filename, '/data', schema='2 * int32')
        dd.extend([(1, 1), (2, 2)])

        results = list(dd)

        self.assertEquals(list(map(list, results)), [[1, 1], [2, 2]])

    def test_schema(self):
        dd = HDF5(self.filename, '/data', schema='2 * int32')

        self.assertEquals(str(dd.schema), '2 * int32')
        self.assertEquals(str(dd.dshape), 'var * 2 * int32')

    def test_dshape(self):
        dd = HDF5(self.filename, '/data', dshape='var * 2 * int32')

        self.assertEquals(str(dd.schema), '2 * int32')
        self.assertEquals(str(dd.dshape), 'var * 2 * int32')

    def test_setitem(self):
        dd = HDF5(self.filename, 'data', dshape='2 * 2 * 2 * int')
        dd[:] = 1
        dd[0, 0, :] = 2
        self.assertEqual(nd.as_py(dd.as_dynd()), [[[2, 2], [1, 1]],
                                                  [[1, 1], [1, 1]]])


class TestIndexing(MakeFile):
    data = [(1, 100),
            (2, 200),
            (3, 300)]

    def test_simple(self):
        dd = HDF5(self.filename, 'data',
                  dshape='var * {x: int, y: int}')
        dd.extend(self.data)

        self.assertEqual(dd[0, 0], 1)
        self.assertEqual(dd[0, 'x'], 1)
        self.assertEqual(tuple(dd[[0, 1], 'x']), (1, 2))
        self.assertEqual(tuple(dd[[0, 1], 'y']), (100, 200))
        self.assertEqual(tuple(dd[::2, 'y']), (100, 300))

    @xfail(reason="when the world improves")
    def test_out_of_order_rows(self):
        assert tuple(dd[[1, 0], 'x']) == (2, 1)

    def test_multiple_fields(self):
        dd = HDF5(self.filename, 'data',
                  dshape='var * {x: int, y: int}')
        dd.extend(self.data)
        self.assertEqual(tuple(dd[[0, 1], ['x', 'y']]), ((1, 100),
                                                            (2, 200)))
        self.assertEqual(into((), dd.dynd[[0, 1], ['x', 'y']]),
                        ((1, 100), (2, 200)))


class TestRecordInputs(MakeFile):

    def test_record_types_chunks(self):
        dd = HDF5(self.filename, 'data', dshape='var * {x: int, y: int}')
        dd.extend_chunks([nd.array([(1, 1), (2, 2)], dtype='{x: int, y: int}')])
        self.assertEqual(tuple(dd), ((1, 1), (2, 2)))

    def test_record_types_extend(self):
        dd = HDF5(self.filename, 'data', dshape='var * {x: int, y: int}')
        dd.extend([(1, 1), (2, 2)])
        self.assertEqual(tuple(dd), ((1, 1), (2, 2)))

    def test_record_types_extend_with_dicts(self):
        dd = HDF5(self.filename, 'data', dshape='var * {x: int, y: int}')
        dd.extend([{'x': 1, 'y': 1}, {'x': 2, 'y': 2}])
        self.assertEqual(tuple(dd), ((1, 1), (2, 2)))


class TestTypes(MakeFile):
    @xfail(reason="h5py doesn't support datetimes well")
    def test_date(self):
        dd = HDF5(self.filename, 'data',
                  dshape='var * {x: int, y: date}')
        dd.extend([(1, date(2000, 1, 1)), (2, date(2000, 1, 2))])

    @xfail(reason="h5py doesn't support datetimes well")
    def test_datetime(self):
        dd = HDF5(self.filename, 'data',
                  dshape='var * {x: int, y: datetime}')
        dd.extend([(1, datetime(2000, 1, 1, 12, 0, 0)),
                   (2, datetime(2000, 1, 2, 12, 30, 00))])


class TestDiscovery(MakeFile):
    def test_discovery(self):
        dd = HDF5(self.filename, 'data',
                  schema='2 * int32')
        dd.extend([(1, 2), (2, 3), (4, 5)])
        with h5py.File(dd.path) as f:
            d = f.get(dd.datapath)
            self.assertEqual(discover(d),
                             dshape('3 * 2 * int32'))

    def test_strings(self):
        schema = '{x: int32, y: string}'
        dd = HDF5(self.filename, 'data',
                  schema=schema)
        dd.extend([(1, 'Hello'), (2, 'World!')])

        with h5py.File(dd.path) as f:
            d = f.get(dd.datapath)
            self.assertEqual(discover(d),
                             dshape('2 * ' + schema))

    def test_ddesc_discovery(self):
        dd = HDF5(self.filename, 'data',
                  schema='2 * int32')
        dd.extend([(1, 2), (2, 3), (4, 5)])
        dd2 = HDF5(self.filename, 'data')

        self.assertEqual(dd.schema, dd2.schema)
        self.assertEqual(dd.dshape, dd2.dshape)

    @xfail(reason="No longer enforcing same dshapes")
    def test_ddesc_conflicts(self):
        dd = HDF5(self.filename, 'data', schema='2 * int32')
        dd.extend([(1, 2), (2, 3), (4, 5)])
        with pytest.raises(TypeError):
            HDF5(self.filename, 'data', schema='2 * float32')
