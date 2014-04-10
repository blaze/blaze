from blaze.datadescriptor import CSV_DDesc, JSON_DDesc
from blaze.datadescriptor.util import filetext
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
