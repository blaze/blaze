import unittest

from dynd import nd
import numpy as np

from blaze.user import validate, into
import blaze


class Test_Validate(unittest.TestCase):
    def test_blaze(self):
        assert validate('2 * int', blaze.array([1, 2]))
        assert not validate('3 * int', blaze.array([1, 2]))

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
        self.assertEqual(into([], blaze.array([1, 2])),
                                  [1, 2])
        self.assertEqual(into((), blaze.array([1, 2])),
                                  (1, 2))
        self.assertEqual(into([], blaze.array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])
        self.assertEqual(into((), blaze.array([[1, 2], [3, 4]])),
                                  ((1, 2), (3, 4)))
        self.assertEqual(into([], into(nd.array(), blaze.array([[1, 2], [3, 4]]))),
                         into([], nd.array([[1, 2], [3, 4]])))
        self.assertEqual(into([], into(blaze.array(0), nd.array([1, 2, 3]))),
                         into([], blaze.array([1, 2, 3])))

    def test_numpy(self):
        assert (into(np.array(0), [1, 2]) == np.array([1, 2])).all()
        self.assertEqual(into([], np.array([1, 2])),
                         [1, 2])
        assert blaze.all(into(blaze.array(0), np.array([1, 2]))
                      == blaze.array([1, 2]))
