from blaze.datadescriptor import CSV_DDesc, JSON_DDesc, HDF5_DDesc
from blaze.datadescriptor.util import filetext, openfile
import json

def test_csv_json():
    with filetext('1,1\n2,2\n') as csv_fn, filetext('') as json_fn:
        schema = '2 * int'
        csv = CSV_DDesc(csv_fn, schema=schema)
        json = JSON_DDesc(json_fn, mode='w', schema=schema)

        json.extend(csv)

        assert list(json) == [[1, 1], [2, 2]]


def test_json_csv_structured():
    data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
    text = '\n'.join(map(json.dumps, data))
    schema = '{x: int, y: int}'

    with filetext(text) as json_fn, filetext('') as csv_fn:
        js = JSON_DDesc(json_fn, schema=schema)
        csv = CSV_DDesc(csv_fn, mode='rw+', schema=schema)

        csv.extend(js)

        assert list(csv) == [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]


def test_csv_json_chunked():
    with filetext('1,1\n2,2\n') as csv_fn, filetext('') as json_fn:
        schema = '2 * int'
        csv = CSV_DDesc(csv_fn, schema=schema)
        json = JSON_DDesc(json_fn, mode='w', schema=schema)

        json.extend_chunks(csv.iterchunks(blen=1))

        assert list(json) == [[1, 1], [2, 2]]


def test_json_csv_chunked():
    data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
    text = '\n'.join(map(json.dumps, data))
    schema = '{x: int, y: int}'

    with filetext(text) as json_fn, filetext('') as csv_fn:
        js = JSON_DDesc(json_fn, schema=schema)
        csv = CSV_DDesc(csv_fn, mode='rw+', schema=schema)

        csv.extend_chunks(js.iterchunks(blen=1))

        assert list(csv) == data

def test_hdf5_csv():
    import h5py
    with openfile('hdf5') as hdf5_fn, filetext('') as csv_fn:
        with h5py.File(hdf5_fn, 'w') as f:
            d = f.create_dataset('data', (3, 3), dtype='i8')
            d[:] = 1

        csv = CSV_DDesc(csv_fn, mode='rw+', schema='3 * int')
        hdf5 = HDF5_DDesc(hdf5_fn, '/data')

        csv.extend_chunks(hdf5.iterchunks(blen=2))

        assert list(csv) == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

"""
def dont_test_csv_hdf5():
    import h5py
    with openfile('hdf5') as hdf5_fn, filetext('1,1\n2,2\n') as csv_fn:
        csv = CSV_DDesc(csv_fn, schema='2 * int')
        hdf5 = HDF5_DDesc(hdf5_fn, '/data', mode='a')

        hdf5.extend_chunks(csv.iterchunks(blen=1))

        assert nd.as_py(hdf5.dynd_arr()) == [[1, 1], [2, 2]]
"""
