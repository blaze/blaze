import json
import numpy as np
import unittest

from blaze.data import CSV, JSON_Streaming, SQL
from blaze.api.into import into
from blaze.utils import filetext, tmpfile
from blaze.data.utils import tuplify
from blaze.h5py import resource

class SingleTestClass(unittest.TestCase):
    def test_csv_json(self):
        with filetext('1,1\n2,2\n') as csv_fn:
            with filetext('') as json_fn:
                schema = '{a: int32, b: int32}'
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
                schema = '{a: int32, b: int32}'
                csv = CSV(csv_fn, schema=schema)
                json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                into(json, csv)

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

                into(csv, js)

                self.assertEquals(tuple(csv), tuples)

    def test_hdf5_csv(self):
        import h5py
        with tmpfile('hdf5') as hdf5_fn:
            with filetext('') as csv_fn:
                with h5py.File(hdf5_fn, 'w') as f:
                    d = f.create_dataset('data', (3,),
                                         dtype=np.dtype([(c, 'i4')
                                                         for c in 'abc']))
                    d[:] = np.array(1)

                csv = CSV(csv_fn, mode='r+', schema='{a: int32, b: int32, c: int32}')
                hdf5 = resource(hdf5_fn + '::/data')

                into(csv, hdf5)

                self.assertEquals(tuple(map(tuple, csv)),
                                  ((1, 1, 1), (1, 1, 1), (1, 1, 1)))

    def test_csv_sql_json(self):
        data = [('Alice', 100), ('Bob', 200)]
        text = '\n'.join(','.join(map(str, row)) for row in data)
        schema = '{name: string, amount: int}'
        with filetext(text) as csv_fn:
            with filetext('') as json_fn:
                with tmpfile('db') as sqldb:

                    csv = CSV(csv_fn, mode='r', schema=schema)
                    sql = SQL('sqlite:///' + sqldb, 'testtable', schema=schema)
                    json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                    into(sql, csv)

                    self.assertEqual(into(list, sql), data)

                    into(json, sql)

                    with open(json_fn) as f:
                        assert 'Alice' in f.read()

    def test_csv_hdf5(self):
        from dynd import nd
        with tmpfile('hdf5') as hdf5_fn:
            with filetext('1,1\n2,2\n') as csv_fn:
                csv = CSV(csv_fn, schema='{a: int32, b: int32}')
                hdf5 = resource(hdf5_fn + '::/data', dshape='var * {a: int32, b: int32}')

                into(hdf5, csv)

                self.assertEquals(hdf5[:].tolist(),
                                  [(1, 1), (2, 2)])
