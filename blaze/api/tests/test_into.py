# encoding: utf8

import unittest

from dynd import nd
import numpy as np
import bcolz
import tables as tb
from datashape import dshape
from datetime import datetime
import os

from blaze.compute.chunks import ChunkIterator, chunks

import pandas as pd
from pandas import DataFrame
from blaze.data import CSV

from blaze.api.into import into, discover
from blaze import Data
from blaze.data import Concat
from blaze.utils import tmpfile, filetext, example
from blaze.pytables import PyTables
import pytest


class TestInto(unittest.TestCase):
    def test_containers(self):
        self.assertEqual(into([], (1, 2, 3)), [1, 2, 3])
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


@pytest.yield_fixture
def h5():
    pytest.importorskip('tables')
    from tables import IsDescription, UInt8Col, StringCol, open_file

    class Test(IsDescription):
        posted_dow = UInt8Col(pos=0)
        jobtype = UInt8Col(pos=1)
        location = UInt8Col(pos=2)
        date = StringCol(20, pos=3)
        country = StringCol(2, pos=4)

    with tmpfile('.h5') as filename:
        h5file = open_file(filename, mode="w", title="Test file")
        group = h5file.create_group("/", 'test', 'Info')
        tab = h5file.create_table(group, 'sample', Test, "Example")

        # Insert a new record
        tab.append([(3, 1, 0, '20121105', 'ab')])
        tab.flush()

        yield h5file
        h5file.close()


data = [('Alice', 100), ('Bob', 200)]
schema = '{name: string, amount: int}'
data_table = Data(data, '2 * {name: string, amount: int}')

@pytest.fixture
def cds():
    pytest.importorskip('bokeh')
    from bokeh.objects import ColumnDataSource
    cds = ColumnDataSource({
     'id': [1, 2, 3],
     'name': ['Alice', 'Bob', 'Charlie'],
     'amount': [100, 200, 300],
     'timestamp': [datetime(2000, 12, 25, 0, 0, 1),
                   datetime(2001, 12, 25, 0, 0, 1),
                   datetime(2002, 12, 25, 0, 0, 1)]
    })
    return cds



@pytest.yield_fixture
def good_csv():
    with tmpfile(".csv") as filename:
        with open(filename, mode='w') as f:
            # Insert a new record
            f.write("userid,text,country\n")
            f.write("1,Alice,az\n")
            f.write("2,Bob,bl\n")
            f.write("3,Charlie,cz\n")
        yield filename


@pytest.yield_fixture
def bad_csv_df():
    with tmpfile(".csv") as filename:
        with open(filename, mode='w') as badfile:
            # Insert a new record
            badfile.write("userid,text,country\n")
            badfile.write("1,Alice,az\n")
            badfile.write("2,Bob,bl\n")
            for i in range(100):
                badfile.write("%d,badguy,zz\n" % i)
            badfile.write("4,Dan,gb,extra,extra\n")
        yield filename


@pytest.yield_fixture
def out_hdf5():
    pytest.importorskip('tables')
    with tmpfile(".h5") as filename:
        yield filename

@pytest.yield_fixture
def out_hdf5_alt():
    pytest.importorskip('tables')
    with tmpfile(".h5") as filename:
        yield filename



class A(object): pass
class B(object): pass

def test_into_fails():
    with pytest.raises(NotImplementedError):
        into(A(), B())


def test_into_pytables_dataframe(h5):
    samp = h5.root.test.sample
    final = into(pd.DataFrame, samp)
    assert len(final) == 1


def test_pandas_dynd():
    arr = nd.array(data, dtype=schema)

    result = into(DataFrame, arr)
    expected = DataFrame(data, columns=['name', 'amount'])
    assert str(result) == str(expected)

    nda = nd.array([[1,2,3], [4,5,6], [7,8,9]])
    csv = CSV(example('accounts.csv'))
    df_csv = into(DataFrame, csv)
    df_nd = into(df_csv, nda)
    df_no_names = into(DataFrame, nda)

    assert list(df_nd.columns) == list(df_csv.columns)
    assert list(df_no_names.columns) == [0,1,2]


def test_pandas_numpy():
    dtype = [('name', 'O'), ('amount', int)]

    x = np.array(data, dtype=dtype)

    result = into(DataFrame(), x)
    expected = DataFrame(data, columns=['name', 'amount'])
    assert str(result) == str(expected)

    result = into(DataFrame(columns=['name', 'amount']), x)
    expected = DataFrame(data, columns=['name', 'amount'])
    assert str(result) == str(expected)


def test_pandas_seq():
    assert str(into(DataFrame, [1, 2])) == str(DataFrame([1, 2]))
    assert str(into(DataFrame, (1, 2))) == str(DataFrame([1, 2]))
    assert (str(into(DataFrame(columns=['a', 'b']), [(1, 2), (3, 4)])) ==
            str(DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])))


def test_pandas_pandas():
    df = DataFrame(data, columns=['name', 'balance'])
    new_df = into(DataFrame, df)
    # Data must be the same
    assert np.all(new_df == df)
    # new_df should be a copy of df
    assert id(new_df) != id(df)


def test_DataFrame_Series():
    df = DataFrame(data, columns=['name', 'balance'])

    new_df = into(DataFrame, df['name'])

    assert np.all(new_df == DataFrame([['Alice'], ['Bob']], columns=['name']))

    # new_df should be a copy of df
    assert id(new_df) != id(df['name'])

    assert isinstance(new_df, DataFrame)


def test_discover_ndarray():
    arr = nd.array(data, dtype=schema)
    assert discover(arr) == 2 * dshape(schema)


def test_discover_pandas():
    df = DataFrame(data, columns=['name', 'balance'])
    assert discover(df).subshape[0] == dshape(schema)


def test_discover_pandas():
    df = DataFrame(data, columns=['name', 'balance'])

    result = into(nd.array, df)

    assert nd.as_py(result, tuple=True) == data


def test_into_table_dataframe():
    t = data_table
    assert list(into(DataFrame(), t).columns) == list(t.fields)
    assert into([], into(DataFrame(), t)) == list(map(tuple, data))


def test_Column_data_source():
    pytest.importorskip('bokeh')
    from bokeh.objects import ColumnDataSource

    cds = into(ColumnDataSource(), data_table)

    assert isinstance(cds, ColumnDataSource)
    assert set(cds.column_names) == set(data_table.fields)


def test_into_ColumnDataSource_pytables():
    pytest.importorskip('bokeh')
    from bokeh.objects import ColumnDataSource

    pyt = PyTables(example('accounts.h5'), '/accounts')
    cds = into(ColumnDataSource, pyt)
    assert 'balance' and 'id' and 'name' in cds.column_names



def test_numpy_list():
    dtype = into(np.ndarray, data).dtype
    assert np.issubdtype(dtype[0], object)
    assert np.issubdtype(dtype[1], int)

    assert into([], into(np.ndarray, data)) == data


def test_numpy_table_expr():
    t = Data(data, 'var * {name: string, amount: int64}')
    assert (into(np.ndarray, t).dtype ==
            np.dtype([('name', 'O'), ('amount', 'i8')]))


def test_DataFrame_CSV():
    with filetext('1,2\n3,4\n') as fn:
        csv = CSV(fn, schema='{a: int64, b: float64}')
        df = into(DataFrame, csv)

        expected = DataFrame([[1, 2.0], [3, 4.0]],
                             columns=['a', 'b'])

        assert str(df) == str(expected)
        assert list(df.dtypes) == [np.int64, np.float64]


def test_into_tables_path(good_csv, out_hdf5, out_hdf5_alt):
    import tables as tb
    tble = into(tb.Table, good_csv, filename=out_hdf5, datapath='/foo')
    tble2 = into(tb.Table, good_csv, filename=out_hdf5_alt, datapath='/foo',
        output_path=out_hdf5_alt)
    n = len(tble)
    x = len(tble2)
    tble._v_file.close()
    assert n == x
    assert n == 3


def test_into_tables_chunk_iterator():
    try:
        pyt = PyTables("foo.h5", "/table", dshape='{x: int32, y: int32}')
        x = np.array([(int(i), int(i)) for i in range(4)], dtype=[('x', np.int32), ('y', np.int32)])
        cs = chunks(x, chunksize=2)
        tble = into(pyt, ChunkIterator(cs))
        n = len(tble)
        tble._v_file.close()
        assert n == 4
    finally:
        try:
            os.remove('foo.h5')
        except OSError:
            pass


def test_into_csv_blaze_table(good_csv):
    t = Data(CSV(good_csv))
    df = into(pd.DataFrame, t[['userid', 'text']])
    assert list(df.columns) == ['userid', 'text']


def test_into_tables_path_bad_csv(bad_csv_df, out_hdf5):
    import tables as tb
    tble = into(tb.Table, bad_csv_df, filename=out_hdf5, datapath='/foo',
                error_bad_lines=False)
    df_from_tbl = into(DataFrame, tble)
    tble._v_file.close()

    # Check that it's the same as straight from the CSV
    df_from_csv = into(DataFrame, bad_csv_df, error_bad_lines=False)
    assert len(df_from_csv) == len(df_from_tbl)
    assert list(df_from_csv.columns) == list(df_from_tbl.columns)
    assert (df_from_csv == df_from_tbl).all().all()


def test_into_ctable_pytables():
    from bcolz import ctable
    tble = PyTables(example('accounts.h5'), datapath='/accounts')
    ct = into(ctable, tble)
    ctn = len(ct)
    tbn = len(tble)
    ctf, ctl = ct[0], ct[-1]
    tbf, tbl = tble[0], tble[-1]
    tble._v_file.close()
    assert ctn == tbn
    assert ctf == tbf
    assert ctl == tbl


def test_into_np_ndarray_carray():
    cr = bcolz.carray([1,2,3,4,5])
    npa = into(np.ndarray, cr)
    assert (npa == cr[:]).all()


def test_into_pd_series_carray():
    cr = bcolz.carray([1,2,3,4,5])
    pda = into(pd.Series, cr)
    assert (pda == cr[:]).all()


def test_numpy_datetimes():
    L = [datetime(2000, 12, 1), datetime(2000, 1, 1, 1, 1, 1)]
    assert into([], np.array(L, dtype='M8[us]')) == L
    assert into([], np.array(L, dtype='M8[ns]')) == L


def test_numpy_python3_bytes_to_string_conversion():
    x = np.array(['a', 'b'], dtype='S1')
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
                    columns=["amount", "id", "name", "timestamp"])
    df = into(DataFrame, fn)
    assert (df == exp).all().all()


def test_into_DataFrame_Excel_xlsx_format():
    pytest.importorskip('xlrd')
    dirname = os.path.dirname(__file__)
    fn = os.path.join(dirname, 'accounts_1.xlsx')
    exp = DataFrame([[1, "Alice", 100],
                     [2, "Bob", 200]],
                    columns=["id", "name", "amount"])
    df = into(DataFrame, fn)
    assert (df == exp).all().all()


def test_into_numpy_from_tableexpr_with_option_types():
    t = Data([[1, 'Alice'], [2, 'Bob']],
              '2 * {id: ?int32, name: string[5, "ascii"]}')
    assert into(np.ndarray, t).dtype == np.dtype([('id', 'i4'), ('name', 'S5')])


def test_into_cds_mixed():
    pytest.importorskip('bokeh')
    from bokeh.objects import ColumnDataSource
    n = 25
    ddict = {'first': np.random.choice(list('abc'), size=n),
             'second': np.random.choice(['cachaça', 'tres leches', 'pizza'],
                                        size=n),
             'third': list(range(n))}
    df = pd.DataFrame(ddict)
    with tmpfile('.csv') as fn:
        df.to_csv(fn, header=None, index=False, encoding='utf8')
        csv = CSV(fn, columns=['first', 'second', 'third'], encoding='utf8')
        t = Data(csv)

        cds = into(ColumnDataSource, t)
        assert isinstance(cds, ColumnDataSource)
        expected = dict((k, into(list, csv[:, k]))
                        for k in ['first', 'second', 'third'])
        assert cds.data == expected

        cds = into(ColumnDataSource, t[['first', 'second']])
        assert isinstance(cds, ColumnDataSource)
        expected = dict((k, into(list, csv[:, k]))
                        for k in ['first', 'second'])
        assert cds.data == expected

        cds = into(ColumnDataSource, t['first'])
        assert isinstance(cds, ColumnDataSource)
        assert cds.data == {'first': into(list, csv[:, 'first'])}


def test_series_single_column():
    data = [('Alice', -200.0, 1), ('Bob', -300.0, 2)]
    t = Data(data, '2 * {name: string, amount: float64, id: int64}')

    df = into(pd.Series, t['name'])
    out_df = into(df, into(DataFrame, t['amount']))
    assert isinstance(df, pd.Series)
    expected = pd.DataFrame(data, columns=t.schema.measure.names).name
    assert str(df) == str(expected)
    assert df.name == out_df.name


def test_into_series_failure():
    failure = into(DataFrame, data)
    with pytest.raises(TypeError):
        into(pd.Series, failure)


def test_series_single_column_projection():
    data = [('Alice', -200.0, 1), ('Bob', -300.0, 2)]
    t = Data(data, '2 * {name: string, amount: float64, id: int64}')
    df = into(pd.Series, t[['name']])
    assert isinstance(df, pd.Series)
    expected = pd.DataFrame(data, columns=t.schema.measure.names).name
    assert str(df) == str(expected)


def test_data_frame_single_column():
    data = [('Alice', -200.0, 1), ('Bob', -300.0, 2)]
    t = Data(data, '2 * {name: string, amount: float64, id: int64}')

    df = into(pd.DataFrame, t['name'])
    assert isinstance(df, pd.DataFrame)
    expected = pd.DataFrame(data, columns=t.schema.measure.names)[['name']]
    assert str(df) == str(expected)


def test_data_frame_single_column_projection():
    data = [('Alice', -200.0, 1), ('Bob', -300.0, 2)]
    t = Data(data, '2 * {name: string, amount: float64, id: int64}')

    df = into(pd.DataFrame, t[['name']])
    assert isinstance(df, pd.DataFrame)
    expected = pd.DataFrame(data, columns=t.schema.measure.names)[['name']]
    assert str(df) == str(expected)


def test_df_from_cds(cds):
    df = into(pd.DataFrame, cds)
    cdsdf = cds.to_df()
    assert (df['name'] == cdsdf['name']).all()


def test_datetime_csv_reader_same_as_into():
    csv = CSV(os.path.join(os.path.dirname(__file__),
                           'accounts.csv'))
    rhs = csv.pandas_read_csv().dtypes
    df = into(pd.DataFrame, csv)
    dtypes = df.dtypes
    expected = pd.Series([np.dtype(x) for x in
                          ['i8', 'i8', 'O', 'datetime64[ns]']],
                         index=csv.columns)
    # make sure reader with no args does the same thing as into()
    # Values the same
    assert dtypes.index.tolist() == rhs.index.tolist()
    assert dtypes.tolist() == rhs.tolist()


@pytest.mark.xfail(reason="pandas reader uses float64 for ?int64")
def test_datetime_csv_reader_same_as_into_types():
    csv = CSV(os.path.join(os.path.dirname(__file__),
                           'accounts.csv'))
    rhs = csv.pandas_read_csv().dtypes
    df = into(pd.DataFrame, csv)
    dtypes = df.dtypes
    expected = pd.Series([np.dtype(x) for x in
                          ['i8', 'i8', 'O', 'datetime64[ns]']],
                         index=csv.columns)
    assert dtypes.index.tolist() == expected.index.tolist()
    assert dtypes.tolist() == expected.tolist()


def test_into_DataFrame_concat():
    csv = CSV(os.path.join(os.path.dirname(__file__),
                           'accounts.csv'))
    df = into(pd.DataFrame, Concat([csv, csv]))
    csv_df = csv.pandas_read_csv()
    assert df.index.tolist() == list(range(len(df)))
    assert df.values.tolist() == (csv_df.values.tolist() +
                                  csv_df.values.tolist())
    assert df.columns.tolist() == csv_df.columns.tolist()


def test_literal_to_literal():
    assert into(10, 10) == 10


def test_into_list_Column():
    with filetext('Alice,1\nBob,2') as fn:
        csv = CSV(fn, columns=['name', 'id'])
        t = Data(csv)
        assert into(list, t.name) == ['Alice', 'Bob']


def test_into_filename():
    with tmpfile('csv') as filename:
        df = DataFrame([['Alice', 100], ['Bob', 200]],
                       columns=['name', 'amount'])
        into(filename, df)

        csv = CSV(filename)
        assert into(list, csv) == into(list, df)


def test_into_filename_filename():
    with filetext('1,2\n3,4', extension='csv') as source_fn:
        with tmpfile('csv') as target_fn:
            into(target_fn, source_fn)

            csv = CSV(target_fn)
            assert into(list, csv) == [(1, 2), (3, 4)]


def test_into_curries():
    assert into(list, (1, 2, 3)) == into(list)((1, 2, 3))


def test_into_df_with_names_from_series():
    df = pd.DataFrame(columns=['a'])
    s = pd.Series([1, 2, 3])
    assert str(pd.DataFrame(s, columns=df.columns)) == str(into(df, s))
    assert str(pd.DataFrame(s, columns=[s.name])) == str(into(pd.DataFrame, s))

    df = pd.DataFrame()
    s = pd.Series([1, 2, 3], name='a')
    assert str(pd.DataFrame(s, columns=['a'])) == str(into(df, s))
    assert str(pd.DataFrame(s, columns=[s.name])) == str(into(pd.DataFrame, s))

    df = pd.DataFrame(columns=['b'])
    s = pd.Series([1, 2, 3], name='a')
    assert str(pd.DataFrame(s, columns=df.columns)) == str(into(df, s))
    assert str(pd.DataFrame(s, columns=[s.name])) == str(into(pd.DataFrame, s))

    with pytest.raises(AssertionError):
        into(pd.DataFrame(columns=list('ab')), s)
