from blaze.data import CSV, JSON

from blaze.utils import tmpfile, raises

import gzip

def test_gzopen_csv():
    with tmpfile('.csv.gz') as filename:
        with gzip.open(filename, 'w') as f:
            f.write('1,1\n2,2')

        # Not a valid CSV file
        assert raises(Exception, lambda: list(CSV(filename, schema='2 * int')))

        dd = CSV(filename, schema='2 * int', open=gzip.open)

        assert list(dd) == [[1, 1], [2, 2]]


def test_gzopen_json():
    with tmpfile('.json.gz') as filename:
        with gzip.open(filename, 'w') as f:
            f.write('[[1, 1], [2, 2]]')

        # Not a valid JSON file
        assert raises(Exception, lambda: list(JSON(filename, schema='2 * int')))

        dd = JSON(filename, schema='2 * int', open=gzip.open)

        assert list(dd) == [[1, 1], [2, 2]]
