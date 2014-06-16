import unittest

from dynd import nd
import numpy as np

from blaze.api import into
import blaze

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
    dd = Python(data, schema='{name: string, amount: int}')
    result = into(DataFrame, dd)
    expected = DataFrame(data, columns=['name', 'amount'])
    print(result)
    print(expected)

    assert str(result) == str(expected)
