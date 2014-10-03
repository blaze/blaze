import os
import pytest
from blaze.api.resource import resource
from blaze.data import CSV, Excel, SQL, HDF5
from blaze.api.into import into

from unittest import TestCase
from blaze.compatibility import xfail
from tempfile import mktemp
import gzip
from blaze.utils import filetext, filetexts, tmpfile

dirname = os.path.dirname(__file__)


def test_resource_csv():
    fn = os.path.join(dirname, 'accounts_1.csv')
    assert isinstance(resource(fn), CSV)


def test_into_resource():
    fn = os.path.join(dirname, 'accounts_1.csv')
    assert into(list, fn) == [(1, 'Alice', 100),
                              (2, 'Bob', 200)]


def test_into_directory_of_csv_files():
    fns = os.path.join(dirname, 'accounts_*.csv')
    assert into(list, fns) == [(1, 'Alice', 100),
                               (2, 'Bob', 200),
                               (3, 'Charlie', 300),
                               (4, 'Dan', 400),
                               (5, 'Edith', 500)]

def test_resource_different_csv_schemas():
    files = {'foobar_a.csv': '1.0,1\n2.0,2',
             'foobar_b.csv': '3,3\n4,4'}
    with filetexts(files):
        r = resource('foobar_*.csv')
        assert r.data[0].schema == r.data[1].schema


def test_into_xls_file():
    pytest.importorskip('xlrd')
    fn = os.path.join(dirname, 'accounts.xls')
    assert isinstance(resource(fn), Excel)


def test_into_xlsx_file():
    pytest.importorskip('xlrd')
    fn = os.path.join(dirname, 'accounts.xlsx')
    assert isinstance(resource(fn), Excel)


def test_into_directory_of_xlsx_files():
    pytest.importorskip('xlrd')
    fns = os.path.join(dirname, 'accounts_*.xlsx')
    assert into(list, fns) == [(1, 'Alice', 100),
                               (2, 'Bob', 200),
                               (3, 'Charlie', 300),
                               (4, 'Dan', 400),
                               (5, 'Edith', 500)]


class TestResource(TestCase):
    def setUp(self):
        self.filename = mktemp()

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_resource_csv(self):
        with filetext('1,1\n2,2', extension='.csv') as fn:
            dd = resource(fn, schema='{x: int, y: int}')
            assert isinstance(dd, CSV)
            self.assertEqual(into(list, dd), [(1, 1), (2, 2)])

    @xfail(os.name.lower() not in ['posix'],
           reason='Windows is hard to please')
    def test_resource_gz(self):
        with filetext(b'1,1\n2,2\n', extension='.csv.gz', open=gzip.open,
                      mode='wb') as fn:
            dd = resource(fn, schema='{x: int, y: int}')
            assert isinstance(dd, CSV)
            assert dd.open == gzip.open
            assert into(list, dd) == [(1, 1), (2, 2)]

    def test_filesystem(self):
        prefix = 'test_filesystem'
        d = {prefix + 'a.csv': '1,1\n2,2',
             prefix + 'b.csv': '1,1\n2,2'}
        with filetexts(d) as filenames:
            dd = resource(prefix + '*.csv', schema='{x: int, y: int}')
            self.assertEqual(into(list, dd),
                            [(1, 1), (2, 2), (1, 1), (2, 2)])

    def test_sql(self):
        with tmpfile('.db') as filename:
            assert isinstance(resource('sqlite:///%s::tablename' % filename,
                                       schema='{x: int, y: int}'),
                              SQL)

    def test_hdf5(self):
        with tmpfile('.hdf5') as filename:
            assert isinstance(resource(filename + '::/path/to/data/',
                                       schema='{a: int, b: int}'),
                              HDF5)


class TestInto(TestCase):
    def test_into(self):
        with filetext('1,1\n2,2', extension='.csv') as a:
            with tmpfile(extension='.csv') as b:
                A = resource(a, schema='{x: int, y: int}')
                B = resource(b, schema='{x: int, y: int}', mode='a')
                B = into(B, A)
                assert into(list, B) == [(1, 1), (2, 2)]

    def test_into_iterable(self):
        with tmpfile(extension='.csv') as fn:
            A = CSV(fn, 'a', schema='{a: int, b: int}')
            data = [(1, 2), (3, 4)]
            A = into(A, data)
            assert list(map(tuple, A)) == list(data)
