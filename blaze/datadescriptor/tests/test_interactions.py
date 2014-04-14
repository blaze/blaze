from blaze.datadescriptor import CSV_DDesc, JSON_DDesc, H5PY_DDesc, copy
from blaze.datadescriptor.util import filetext, openfile
import json
import unittest

class SingleTestClass(unittest.TestCase):
    def test_csv_json(self):
        with filetext('1,1\n2,2\n') as csv_fn:
            with filetext('') as json_fn:
                schema = '2 * int'
                csv = CSV_DDesc(csv_fn, schema=schema)
                json = JSON_DDesc(json_fn, mode='w', schema=schema)

                json.extend(csv)

                self.assertEquals(list(json), [[1, 1], [2, 2]])


    def test_json_csv_structured(self):
        data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
        text = '\n'.join(map(json.dumps, data))
        schema = '{x: int, y: int}'

        with filetext(text) as json_fn:
            with filetext('') as csv_fn:
                js = JSON_DDesc(json_fn, schema=schema)
                csv = CSV_DDesc(csv_fn, mode='r+', schema=schema)

                csv.extend(js)

                self.assertEquals(list(csv),
                                  [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}])


    def test_csv_json_chunked(self):
        with filetext('1,1\n2,2\n') as csv_fn:
            with filetext('') as json_fn:
                schema = '2 * int'
                csv = CSV_DDesc(csv_fn, schema=schema)
                json = JSON_DDesc(json_fn, mode='w', schema=schema)

                copy(csv, json)

                self.assertEquals(list(json), [[1, 1], [2, 2]])


    def test_json_csv_chunked(self):
        data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
        text = '\n'.join(map(json.dumps, data))
        schema = '{x: int, y: int}'

        with filetext(text) as json_fn:
            with filetext('') as csv_fn:
                js = JSON_DDesc(json_fn, schema=schema)
                csv = CSV_DDesc(csv_fn, mode='r+', schema=schema)

                copy(js, csv)

                self.assertEquals(list(csv), data)

    def test_hdf5_csv(self):
        import h5py
        with openfile('hdf5') as hdf5_fn:
            with filetext('') as csv_fn:
                with h5py.File(hdf5_fn, 'w') as f:
                    d = f.create_dataset('data', (3, 3), dtype='i8')
                    d[:] = 1

                csv = CSV_DDesc(csv_fn, mode='r+', schema='3 * int')
                hdf5 = H5PY_DDesc(hdf5_fn, '/data')

                copy(hdf5, csv)

                self.assertEquals(list(csv), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    """
    def dont_test_csv_hdf5(self):
        import h5py
        with openfile('hdf5') as hdf5_fn:
            with filetext('1,1\n2,2\n') as csv_fn:
                csv = CSV_DDesc(csv_fn, schema='2 * int')
                hdf5 = H5PY_DDesc(hdf5_fn, '/data', mode='a')

                copy(csv, hdf5)

                self.assertEquals(nd.as_py(hdf5.dynd_arr()),
                                  [[1, 1], [2, 2]])
    """
