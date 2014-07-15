import unittest

from dynd import nd
import numpy as np
from datashape import dshape

from blaze.api.into import into, discover
import blaze


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
