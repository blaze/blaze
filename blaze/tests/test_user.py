import unittest

from dynd import nd

from blaze.user import validate, into
from blaze import array


class Test_Validate(unittest.TestCase):
    def test_blaze(self):
        assert validate('2 * int', array([1, 2]))
        assert not validate('3 * int', array([1, 2]))

    def test_dynd(self):
        assert validate('2 * int', nd.array([1, 2], dtype='2 * int32'))
        assert not validate('3 * int', nd.array([1, 2], dtype='2 * int32'))

class Test_Into(unittest.TestCase):
    def test_containers(self):
        self.assertEqual(into([], (1, 2, 3)),
                                  [1, 2, 3])
        self.assertEqual(into((), (1, 2, 3)),
                                  (1, 2, 3))
        self.assertEqual(into({}, [(1, 2), (3, 4)]),
                                  {1: 2, 3: 4})
        self.assertEqual(into((), {1: 2, 3: 4}),
                                  ((1, 2), (3, 4)))

    def test_dynd(self):
        self.assertEqual(nd.as_py(into(nd.array(), (1, 2, 3))),
                         nd.as_py(nd.array([1, 2, 3])))
        self.assertEqual(into([], nd.array([1, 2])),
                                  [1, 2])
        self.assertEqual(into([], nd.array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])

    def test_blaze(self):
        self.assertEqual(into([], array([1, 2])),
                                  [1, 2])
        self.assertEqual(into((), array([1, 2])),
                                  (1, 2))
        self.assertEqual(into([], array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])
        self.assertEqual(into((), array([[1, 2], [3, 4]])),
                                  ((1, 2), (3, 4)))
        self.assertEqual(nd.as_py(into(nd.array(), array([[1, 2], [3, 4]]))),
                         nd.as_py(nd.array([[1, 2], [3, 4]])))
