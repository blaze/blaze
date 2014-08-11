import unittest

from dynd import nd
import numpy as np
from datashape import dshape
import tempfile
import os

from blaze.api.into import into, discover
import blaze
from blaze import Table, TableSymbol, compute


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
    from blaze.data.python import Python
except ImportError:
    Python = None

try:
    from bokeh.objects import ColumnDataSource
except ImportError:
    ColumnDataSource = None

try:
    from tables import (Table as PyTables, open_file, UInt8Col, StringCol,
        IsDescription)
except ImportError:
    PyTables = None


class Test_into_pytables(unittest.TestCase):

    @skip_if_not(PyTables and DataFrame)
    def setUp(self):
        self.h5_file = tempfile.mktemp(".h5")
        class Test(IsDescription):
            posted_dow = UInt8Col()
            jobtype  = UInt8Col()
            location  = UInt8Col()
            date  = StringCol(20)
            country  = StringCol(2)

        h5file = open_file(self.h5_file, mode = "w", title = "Test file")
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
        # Close (and flush) the file
        h5file.close()

    @skip_if_not(PyTables and DataFrame)
    def tearDown(self):
        if os.path.exists(self.h5_file):
            os.remove(self.h5_file)

    @skip_if_not(PyTables and DataFrame)
    def test_into_pytables_dataframe(self):
        schema = '''
           {country: string,
           date: string,
           type: uint8,
           location: uint8,
           posted_dow: uint8}
          '''

        thefile = open_file(self.h5_file)
        samp = thefile.root.test.sample
        # Create a Blaze TableSymbol of similar schema
        t = TableSymbol('example', schema=schema)
        final = into(DataFrame, compute(t, {t:samp}))
        self.assertEqual(len(final), 1)

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
    assert into([], into(DataFrame(), t)) == data


@skip_if_not(Table and ColumnDataSource)
def test_Column_data_source():
    data = [('Alice', 100), ('Bob', 200)]
    t = Table(data, columns=['name', 'balance'])

    cds = into(ColumnDataSource(), t)

    assert isinstance(cds, ColumnDataSource)
    assert set(cds.column_names) == set(t.columns)
