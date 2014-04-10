from blaze import array
import blaze

import unittest

class Test_1D_Array(unittest.TestCase):
    def setUp(self):
        self.a = array([1, 2, 3])

    def test_iter_1d(self):
        self.assertEqual(list(self.a), [1, 2, 3])

    def test_get_element(self):
        self.assertEqual(self.a[1], 2)

    def test_get_element(self):
        assert blaze.all(self.a[1:] == array([2, 3]))


class Test_2D_Array(unittest.TestCase):
    def setUp(self):
        self.a = array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

    def test_list_list(self):
        self.assertEqual(list(list(self.a)[0]), [1, 2, 3])

    def test_list_elements(self):
        assert blaze.all(list(self.a)[0] == array([1, 2, 3]))
        assert blaze.all(list(self.a)[1] == array([4, 5, 6]))
