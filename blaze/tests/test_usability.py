import unittest

from dynd import nd
import numpy as np

from blaze.usability import validate, like
import blaze


class Test_Validate(unittest.TestCase):
    def test_blaze(self):
        assert validate('2 * int', blaze.array([1, 2]))
        assert not validate('3 * int', blaze.array([1, 2]))

    def test_dynd(self):
        assert validate('2 * int', nd.array([1, 2], dtype='2 * int32'))
        assert not validate('3 * int', nd.array([1, 2], dtype='2 * int32'))

class Test_Like(unittest.TestCase):
    def test_containers(self):
        self.assertEqual(like([], (1, 2, 3)),
                                  [1, 2, 3])
        self.assertEqual(like((), (1, 2, 3)),
                                  (1, 2, 3))
        self.assertEqual(like({}, [(1, 2), (3, 4)]),
                                  {1: 2, 3: 4})
        self.assertEqual(like((), {1: 2, 3: 4}),
                                  ((1, 2), (3, 4)))
        self.assertEqual(like((), {'cat': 2, 'dog': 4}),
                                  (('cat', 2), ('dog', 4)))

    def test_dynd(self):
        self.assertEqual(nd.as_py(like(nd.array(), (1, 2, 3))),
                         nd.as_py(nd.array([1, 2, 3])))
        self.assertEqual(like([], nd.array([1, 2])),
                                  [1, 2])
        self.assertEqual(like([], nd.array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])

    def test_blaze(self):
        self.assertEqual(like([], blaze.array([1, 2])),
                                  [1, 2])
        self.assertEqual(like((), blaze.array([1, 2])),
                                  (1, 2))
        self.assertEqual(like([], blaze.array([[1, 2], [3, 4]])),
                                  [[1, 2], [3, 4]])
        self.assertEqual(like((), blaze.array([[1, 2], [3, 4]])),
                                  ((1, 2), (3, 4)))
        self.assertEqual(like([], like(nd.array(), blaze.array([[1, 2], [3, 4]]))),
                         like([], nd.array([[1, 2], [3, 4]])))
        self.assertEqual(like([], like(blaze.array(0), nd.array([1, 2, 3]))),
                         like([], blaze.array([1, 2, 3])))

    def test_numpy(self):
        assert (like(np.array(0), [1, 2]) == np.array([1, 2])).all()
        self.assertEqual(like([], np.array([1, 2])),
                         [1, 2])
        assert blaze.all(like(blaze.array(0), np.array([1, 2]))
                      == blaze.array([1, 2]))
