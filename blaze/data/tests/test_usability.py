from unittest import TestCase
import os
from tempfile import mktemp
import gzip

from blaze.utils import filetext, filetexts, tmpfile
from blaze.data.utils import tuplify
from blaze.data import *
from blaze.api.resource import *
from blaze.compatibility import xfail
from blaze.api.into import *


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
            self.assertEqual(tuplify(list(dd)), ((1, 1), (2, 2)))

    @xfail
    def test_resource_gz(self):
        with filetext('1,1\n2,2', extension='.csv.gz', open=gzip.open) as fn:
            dd = resource(fn, schema='2 * int')
            assert isinstance(dd, CSV)
            self.assertEqual(dd.open, gzip.open)
            self.assertEqual(tuplify(list(dd)), ((1, 1), (2, 2)))

    def test_filesystem(self):
        prefix = 'test_filesystem'
        d = {prefix + 'a.csv': '1,1\n2,2',
             prefix + 'b.csv': '1,1\n2,2'}
        with filetexts(d) as filenames:
            dd = resource(prefix + '*.csv', schema='2 * int')
            self.assertEqual(into(list, dd),
                            [(1, 1), (2, 2), (1, 1), (2, 2)])

    def test_sql(self):
        assert isinstance(resource('sqlite:///:memory:::tablename',
                                   schema='{x: int, y: int}'),
                          SQL)

    def test_hdf5(self):
        with tmpfile('.hdf5') as filename:
            assert isinstance(resource(filename + '::/path/to/data/',
                                       schema='2 * int'),
                              HDF5)



class TestInto(TestCase):
    def test_into(self):
        with filetext('1,1\n2,2', extension='.csv') as a:
            with tmpfile(extension='.csv') as b:
                A = resource(a, schema='2 * int')
                B = resource(b, schema='2 * int', mode='a')
                B = into(B, A)
                assert tuplify(list(B)) == ((1, 1), (2, 2))

    def test_into_iterable(self):
        with tmpfile(extension='.csv') as fn:
            A = CSV(fn, 'a', schema='2 * int')
            data = [(1, 2), (3, 4)]
            A = into(A, data)
            assert tuplify(list(A)) == tuplify(data)
