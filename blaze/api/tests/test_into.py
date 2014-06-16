import unittest

from dynd import nd
import numpy as np

from blaze.api import into
import blaze


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
