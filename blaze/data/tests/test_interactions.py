import json
import unittest
from sqlalchemy import create_engine

from blaze.data import CSV, JSON_Streaming, HDF5, SQL, copy
from blaze.utils import filetext, tmpfile
from blaze.data.utils import tuplify

class SingleTestClass(unittest.TestCase):
    def test_csv_json(self):
        with filetext('1,1\n2,2\n') as csv_fn:
            with filetext('') as json_fn:
                schema = '2 * int'
                csv = CSV(csv_fn, schema=schema)
                json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                json.extend(csv)

                self.assertEquals(tuple(map(tuple, json)), ((1, 1), (2, 2)))


    def test_json_csv_structured(self):
        data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
        text = '\n'.join(map(json.dumps, data))
        schema = '{x: int, y: int}'

        with filetext(text) as json_fn:
            with filetext('') as csv_fn:
                js = JSON_Streaming(json_fn, schema=schema)
                csv = CSV(csv_fn, mode='r+', schema=schema)

                csv.extend(js)

                self.assertEquals(tuple(map(tuple, (csv))),
                                  ((1, 1), (2, 2)))


    def test_csv_json_chunked(self):
        with filetext('1,1\n2,2\n') as csv_fn:
            with filetext('') as json_fn:
                schema = '2 * int'
                csv = CSV(csv_fn, schema=schema)
                json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                copy(csv, json)

                self.assertEquals(tuplify(tuple(json)), ((1, 1), (2, 2)))


    def test_json_csv_chunked(self):
        data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
        tuples = ((1, 1), (2, 2))
        text = '\n'.join(map(json.dumps, data))
        schema = '{x: int, y: int}'

        with filetext(text) as json_fn:
            with filetext('') as csv_fn:
                js = JSON_Streaming(json_fn, schema=schema)
                csv = CSV(csv_fn, mode='r+', schema=schema)

                copy(js, csv)

                self.assertEquals(tuple(csv), tuples)

    def test_hdf5_csv(self):
        import h5py
        with tmpfile('hdf5') as hdf5_fn:
            with filetext('') as csv_fn:
                with h5py.File(hdf5_fn, 'w') as f:
                    d = f.create_dataset('data', (3, 3), dtype='i8')
                    d[:] = 1

                csv = CSV(csv_fn, mode='r+', schema='3 * int')
                hdf5 = HDF5(hdf5_fn, '/data')

                copy(hdf5, csv)

                self.assertEquals(tuple(map(tuple, csv)),
                                  ((1, 1, 1), (1, 1, 1), (1, 1, 1)))

    def test_csv_sql_json(self):
        data = [('Alice', 100), ('Bob', 200)]
        text = '\n'.join(','.join(map(str, row)) for row in data)
        schema = '{name: string, amount: int}'
        engine = create_engine('sqlite:///:memory:')
        with filetext(text) as csv_fn:
            with filetext('') as json_fn:

                csv = CSV(csv_fn, mode='r', schema=schema)
                sql = SQL(engine, 'testtable', schema=schema)
                json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                copy(csv, sql)

                self.assertEqual(list(sql), data)

                copy(sql, json)

                with open(json_fn) as f:
                    assert 'Alice' in f.read()

    def test_csv_hdf5(self):
        import h5py
        from dynd import nd
        with tmpfile('hdf5') as hdf5_fn:
            with filetext('1,1\n2,2\n') as csv_fn:
                csv = CSV(csv_fn, schema='2 * int')
                hdf5 = HDF5(hdf5_fn, '/data', schema='2 * int')

                copy(csv, hdf5)

                self.assertEquals(nd.as_py(hdf5.as_dynd()),
                                  [[1, 1], [2, 2]])
