from unittest import TestCase
import os
from tempfile import mktemp
import gzip

from blaze.utils import filetext, filetexts, tmpfile
from blaze.data import *
from blaze.compatibility import skip

class TestResource(TestCase):
    def setUp(self):
        self.filename = mktemp()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_resource_csv(self):
        with filetext('1,1\n2,2', extension='.csv') as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, CSV)
            self.assertEqual(list(dd), [(1, 1), (2, 2)])

    def test_resource_json(self):
        with filetext('[[1,1], [2,2]]', extension='.json') as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, JSON)
            self.assertEqual(list(dd), [(1, 1), (2, 2)])

    def test_resource_gz(self):
        with filetext('1,1\n2,2', extension='.csv.gz', open=gzip.open) as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, CSV)
            self.assertEqual(dd.open, gzip.open)
            self.assertEqual(list(dd), [(1, 1), (2, 2)])

    def test_filesystem(self):
        d = {'a.csv': '1,1\n2,2', 'b.csv': '1,1\n2,2'}
        with filetexts(d) as filenames:
            dd = resource('*.csv', schema='2 * int')
            assert isinstance(dd, Files)

    def test_sql(self):
        assert isinstance(resource('sqlite:///:memory:::tablename',
                                   schema='{x: int, y: int}'),
                          SQL)

    @skip("This runs fine in isolation, segfaults in full test")
    def test_hdf5(self):
        with tmpfile('.hdf5') as filename:
            assert isinstance(resource(filename + '::/path/to/data/',
                                        mode='w', schema='2 * int'),
                              HDF5)

class TestCopy(TestCase):
    def test_copy(self):
        with filetext('1,1\n2,2', extension='.csv') as a:
            with tmpfile(extension='.csv') as b:
                A = resource(a, schema='2 * int')
                B = resource(b, schema='2 * int', mode='a')
                copy(A, B)
                assert list(B) == [(1, 1), (2, 2)]
