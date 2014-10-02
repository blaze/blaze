import json

from blaze import resource
from blaze.data import JSON_Streaming, HDF5
from blaze.api.into import into
from blaze.utils import filetext, tmpfile
from blaze.data.utils import tuplify


def test_csv_json():
    with filetext('1,1\n2,2\n', '.csv') as csv_fn:
        with filetext('') as json_fn:
            schema = '2 * int'
            csv = resource(csv_fn, schema=schema)
            json = JSON_Streaming(json_fn, mode='r+', schema=schema)

            json.extend(csv)

            assert tuple(map(tuple, json)) == ((1, 1), (2, 2))


def test_json_csv_structured():
    data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
    text = '\n'.join(map(json.dumps, data))
    schema = '{x: int, y: int}'

    with filetext(text) as json_fn:
        with filetext('', '.csv') as csv_fn:
            js = JSON_Streaming(json_fn, schema=schema)
            csv = resource(csv_fn, mode='r+', schema=schema)

            csv.extend(js)

            assert tuple(map(tuple, (csv))) == ((1, 1), (2, 2))


def test_csv_json_chunked():
    with filetext('1,1\n2,2\n', '.csv') as csv_fn:
        with filetext('') as json_fn:
            schema = '2 * int'
            csv = resource(csv_fn, schema=schema)
            json = JSON_Streaming(json_fn, mode='r+', schema=schema)

            into(json, csv)

            assert tuplify(tuple(json)) == ((1, 1), (2, 2))


def test_json_csv_chunked():
    data = [{'x': 1, 'y': 1}, {'x': 2, 'y': 2}]
    tuples = ((1, 1), (2, 2))
    text = '\n'.join(map(json.dumps, data))
    schema = '{x: int, y: int}'

    with filetext(text) as json_fn:
        with filetext('', '.csv') as csv_fn:
            js = JSON_Streaming(json_fn, schema=schema)
            csv = resource(csv_fn, mode='r+', schema=schema)

            into(csv, js)

            assert tuple(csv) == tuples


def test_hdf5_csv():
    import h5py
    with tmpfile('hdf5') as hdf5_fn:
        with filetext('', '.csv') as csv_fn:
            with h5py.File(hdf5_fn, 'w') as f:
                d = f.create_dataset('data', (3, 3), dtype='i8')
                d[:] = 1

            csv = resource(csv_fn, mode='r+', schema='3 * int')
            hdf5 = HDF5(hdf5_fn, '/data')

            into(csv, hdf5)

            assert (tuple(map(tuple, csv)) ==
                    ((1, 1, 1), (1, 1, 1), (1, 1, 1)))


def test_csv_sql_json():
    data = [('Alice', 100), ('Bob', 200)]
    text = '\n'.join(','.join(map(str, row)) for row in data)
    schema = '{name: string, amount: int}'
    with filetext(text, '.csv') as csv_fn:
        with filetext('') as json_fn:
            with tmpfile('db') as sqldb:

                csv = resource(csv_fn, mode='r', schema=schema)
                sql = resource('sqlite:///' + sqldb, 'testtable',
                                schema=schema)
                json = JSON_Streaming(json_fn, mode='r+', schema=schema)

                into(sql, csv)

                assert into(list, sql) == data

                into(json, sql)

                with open(json_fn) as f:
                    assert 'Alice' in f.read()


def test_csv_hdf5():
    from dynd import nd

    with tmpfile('hdf5') as hdf5_fn:
        with filetext('1,1\n2,2\n', '.csv') as csv_fn:
            csv = resource(csv_fn, schema='2 * int')
            hdf5 = HDF5(hdf5_fn, '/data', schema='2 * int')

            into(hdf5, csv)

            assert nd.as_py(hdf5.as_dynd()) == [[1, 1], [2, 2]]
