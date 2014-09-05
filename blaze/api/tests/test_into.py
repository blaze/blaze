import unittest

from dynd import nd
import numpy as np
from datashape import dshape
from datetime import datetime
import os

from blaze.api.into import into, discover
from blaze import Table
from blaze.utils import tmpfile
import pytest

def skip(test_foo):
    return

def skip_if_not(x):
    def maybe_a_test_function(test_foo):
        if not x:
            return
        else:
            return test_foo
    return maybe_a_test_function


class Test_into(unittest.TestCase):
    def test_containers(self):
        self.assertEqual(into([], (1, 2, 3)),
                                  [1, 2, 3])
        self.assertEqual(into([], iter((1, 2, 3))),
                                  [1, 2, 3])
        self.assertEqual(into((), (1, 2, 3)),
                                  (1, 2, 3))
        self.assertEqual(into({}, [(1, 2), (3, 4)]),
                                  {1: 2, 3: 4})
        self.assertEqual(into((), {1: 2, 3: 4}),
                                  ((1, 2), (3, 4)))
        self.assertEqual(into((), {'cat': 2, 'dog': 4}),
                                  (('cat', 2), ('dog', 4)))

    def test_dynd(self):
        self.assertEqual(nd.as_py(into(nd.array(), (1, 2, 3))),
                         nd.as_py(nd.array([1, 2, 3])))
        self.assertEqual(into([], nd.array([1, 2])),
                                  [1, 2])
        self.assertEqual(into([], nd.array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])

    def test_numpy(self):
        assert (into(np.array(0), [1, 2]) == np.array([1, 2])).all()
        assert (into(np.array(0), iter([1, 2])) == np.array([1, 2])).all()
        self.assertEqual(into([], np.array([1, 2])),
                         [1, 2])

    def test_numpy_datetime(self):
        assert isinstance(into(np.ndarray(0), [datetime(2014, 1, 1)])[0],
                          np.datetime64)

    def test_type(self):
        self.assertEqual(into(list, (1, 2, 3)),
                         into([], (1, 2, 3)))
        self.assertEqual(str(into(np.ndarray, (1, 2, 3))),
                         str(into(np.ndarray(()), (1, 2, 3))))


try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None

try:
    from blaze.data import Python, CSV
except ImportError:
    Python = None
    CSV = None

try:
    from bokeh.objects import ColumnDataSource
except ImportError:
    ColumnDataSource = None

try:
    from tables import (Table as PyTables, open_file, UInt8Col, StringCol,
        IsDescription)
except ImportError:
    PyTables = None

@pytest.yield_fixture
def h5():
    class Test(IsDescription):
        posted_dow = UInt8Col()
        jobtype  = UInt8Col()
        location  = UInt8Col()
        date  = StringCol(20)
        country  = StringCol(2)

    with tmpfile('.h5') as filename:
        h5file = open_file(filename, mode = "w", title = "Test file")
        group = h5file.create_group("/", 'test', 'Info')
        tab = h5file.create_table(group, 'sample', Test, "Example")
        arow = tab.row
        arow['country'] = 'ab'
        arow['date'] = '20121105'
        arow['location'] = 0
        arow['jobtype']  = 1
        arow['posted_dow']  = 3
        # Insert a new record
        arow.append()
        tab.flush()
        yield h5file
        # Close (and flush) the file
        h5file.close()


@pytest.yield_fixture
def good_csv():

    with tmpfile(".csv") as filename:
        badfile = open(filename, mode="w")
        # Insert a new record
        badfile.write("userid,text,country\n")
        badfile.write("1,Alice,az\n")
        badfile.write("2,Bob,bl\n")
        badfile.write("3,Charlie,cz\n")
        badfile.flush()
        yield badfile
        # Close (and flush) the file
        badfile.close()


@pytest.yield_fixture
def bad_csv_df():

    with tmpfile(".csv") as filename:
        badfile = open(filename, mode="w")
        # Insert a new record
        badfile.write("userid,text,country\n")
        badfile.write("1,Alice,az\n")
        badfile.write("2,Bob,bl\n")
        for i in range(0,100):
            badfile.write(str(i) + ",badguy,zz\n")
        badfile.write("4,Dan,gb,extra,extra\n")
        badfile.flush()
        yield badfile
        # Close (and flush) the file
        badfile.close()

@pytest.yield_fixture
def out_hdf5():
    with tmpfile(".h5") as filename:
        yield filename

@skip_if_not(PyTables and DataFrame)
def test_into_pytables_dataframe(h5):
    samp = h5.root.test.sample
    final = into(DataFrame, samp)
    assert len(final) == 1

@skip_if_not(DataFrame and Python)
def test_pandas_data_descriptor():
    data = [['Alice', 100], ['Bob', 200]]
    schema='{name: string, amount: int}'
    dd = Python(data, schema=schema)
    result = into(DataFrame, dd)
    expected = DataFrame(data, columns=['name', 'amount'])
    print(result)
    print(expected)

    assert str(result) == str(expected)


@skip_if_not(DataFrame and nd.array)
def test_pandas_dynd():
    data = [['Alice', 100], ['Bob', 200]]
    schema='{name: string, amount: int}'

    arr = nd.array(data, dtype=schema)

    result = into(DataFrame, arr)
    expected = DataFrame(data, columns=['name', 'amount'])
    print(result)
    print(expected)

    assert str(result) == str(expected)


@skip_if_not(DataFrame)
def test_pandas_numpy():
    data = [('Alice', 100), ('Bob', 200)]
    dtype=[('name', 'O'), ('amount', int)]

    x = np.array(data, dtype=dtype)

    result = into(DataFrame(), x)
    expected = DataFrame(data, columns=['name', 'amount'])
    assert str(result) == str(expected)

    result = into(DataFrame(columns=['name', 'amount']), x)
    expected = DataFrame(data, columns=['name', 'amount'])
    print(result)
    print(expected)
    assert str(result) == str(expected)


@skip_if_not(DataFrame)
def test_pandas_seq():
    assert str(into(DataFrame, [1, 2])) == \
            str(DataFrame([1, 2]))
    assert str(into(DataFrame, (1, 2))) == \
            str(DataFrame([1, 2]))
    assert str(into(DataFrame(columns=['a', 'b']), [(1, 2), (3, 4)])) == \
            str(DataFrame([[1, 2], [3, 4]], columns=['a', 'b']))


@skip_if_not(DataFrame)
def test_pandas_pandas():
    data = [('Alice', 100), ('Bob', 200)]
    df = DataFrame(data, columns=['name', 'balance'])
    new_df = into(DataFrame, df)
    # Data must be the same
    assert np.all(new_df == df)
    # new_df should be a copy of df
    assert id(new_df) != id(df)


@skip_if_not(DataFrame)
def test_DataFrame_Series():
    data = [('Alice', 100), ('Bob', 200)]
    df = DataFrame(data, columns=['name', 'balance'])

    new_df = into(DataFrame, df['name'])

    assert np.all(new_df == DataFrame([['Alice'], ['Bob']], columns=['name']))

    # new_df should be a copy of df
    assert id(new_df) != id(df['name'])

    assert isinstance(new_df, DataFrame)


def test_discover_ndarray():
    data = [['Alice', 100], ['Bob', 200]]
    schema='{name: string, balance: int32}'
    arr = nd.array(data, dtype=schema)
    assert discover(arr) == 2 * dshape(schema)

@skip_if_not(DataFrame)
def test_discover_pandas():
    data = [['Alice', 100], ['Bob', 200]]
    df = DataFrame(data, columns=['name', 'balance'])

    print(discover(df))
    assert discover(df).subshape[0] == dshape('{name: string, balance: int64}')


@skip_if_not(DataFrame and nd.array)
def test_discover_pandas():
    data = [('Alice', 100), ('Bob', 200)]
    df = DataFrame(data, columns=['name', 'balance'])

    result = into(nd.array, df)

    assert nd.as_py(result, tuple=True) == data

@skip_if_not(Table and DataFrame)
def test_into_table_dataframe():
    data = [['Alice', 100], ['Bob', 200]]
    t = Table(data, columns=['name', 'amount'])

    assert list(into(DataFrame(), t).columns) == list(t.columns)
    assert into([], into(DataFrame(), t)) == list(map(tuple, data))


@skip_if_not(Table and ColumnDataSource)
def test_Column_data_source():
    data = [('Alice', 100), ('Bob', 200)]
    t = Table(data, columns=['name', 'balance'])

    cds = into(ColumnDataSource(), t)

    assert isinstance(cds, ColumnDataSource)
    assert set(cds.column_names) == set(t.columns)


def test_numpy_list():
    data = [('Alice', 100), ('Bob', 200)]

    dtype = into(np.ndarray, data).dtype
    assert np.issubdtype(dtype[0], str)
    assert np.issubdtype(dtype[1], int)

    assert into([], into(np.ndarray, data)) == data


@skip_if_not(Table)
def test_numpy_tableExpr():
    data = [('Alice', 100), ('Bob', 200)]
    t = Table(data, '{name: string, amount: int64}')

    assert into(np.ndarray, t).dtype == \
            np.dtype([('name', 'O'), ('amount', 'i8')])


@skip_if_not(DataFrame and CSV)
def test_DataFrame_CSV():
    with filetext('1,2\n3,4\n') as fn:
        csv = CSV(fn, schema='{a: int64, b: float64}')
        df = into(DataFrame, csv)

        expected = DataFrame([[1, 2.0], [3, 4.0]],
                             columns=['a', 'b'])

        assert str(df) == str(expected)
        assert list(df.dtypes) == [np.int64, np.float64]


@skip_if_not(PyTables and DataFrame)
def test_into_tables_path(good_csv, out_hdf5):
    tble = into(PyTables, good_csv.name, filename=out_hdf5, datapath='foo')
    assert len(tble) == 3

    tble.close()


@skip_if_not(PyTables and DataFrame)
def test_into_tables_path_bad_csv(bad_csv_df, out_hdf5):
    tble = into(PyTables, bad_csv_df.name,
                          filename=out_hdf5,
                          datapath='foo',
                          error_bad_lines=False)
    df_from_tbl = into(DataFrame, tble)
    #Check that it's the same as straight from the CSV
    df_from_csv = into(DataFrame, bad_csv_df.name, error_bad_lines=False)
    assert len(df_from_csv) == len(df_from_tbl)
    assert list(df_from_csv.columns) == list(df_from_tbl.columns)
    assert (df_from_csv == df_from_tbl).all().all()

    tble.close()


def test_numpy_datetimes():
    L = [datetime(2000, 12, 1), datetime(2000, 1, 1, 1, 1, 1)]
    assert into([], np.array(L, dtype='M8[us]')) == L
    assert into([], np.array(L, dtype='M8[ns]')) == L


def test_numpy_python3_bytes_to_string_conversion():
    x = np.array(['a','b'], dtype='S1')
    assert all(isinstance(s, str) for s in into(list, x))
    x = np.array([(1, 'a'), (2, 'b')], dtype=[('id', 'i4'), ('letter', 'S1')])
    assert isinstance(into(list, x)[0][1], str)


def test_into_DataFrame_Excel_xls_format():
    pytest.importorskip('xlrd')
    dirname = os.path.dirname(__file__)
    fn = os.path.join(dirname, 'accounts.xls')
    exp = DataFrame([[100, 1, "Alice", "2000-12-25T00:00:01"],
                    [200, 2, "Bob", "2001-12-25T00:00:01"],
                    [300, 3, "Charlie", "2002-12-25T00:00:01"]], 
                    columns = ["amount", "id", "name", "timestamp"])
    df = into(DataFrame, fn)
    assert (df == exp).all().all()


def test_into_DataFrame_Excel_xlsx_format():
    pytest.importorskip('xlrd')
    dirname = os.path.dirname(__file__)
    fn = os.path.join(dirname, 'accounts_1.xlsx')
    exp = DataFrame([[1, "Alice", 100],
                     [2, "Bob", 200]],
                    columns = ["id", "name", "amount"])
    df = into(DataFrame, fn)
    assert (df == exp).all().all()


def test_into_numpy_from_tableexpr_with_option_types():
    t = Table([[1, 'Alice'], [2, 'Bob']],
              schema='{id: ?int32, name: string[5, "ascii"]}')
    assert into(np.ndarray, t).dtype == \
            np.dtype([('id', 'i4'), ('name', 'S5')])
