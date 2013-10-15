import blaze
from blaze.datadescriptor import dd_as_py
import unittest


class TestBasic(unittest.TestCase):

    def test_add(self):
        types = ['int8', 'int16', 'int32', 'int64']
        for type_ in types:
            a = blaze.array(range(3), dshape=type_)
            c = blaze.eval(a+a)
            self.assertEqual(dd_as_py(c._data), [0, 2, 4])
            c = blaze.eval(((a+a)*a))
            self.assertEqual(dd_as_py(c._data), [0, 2, 8])

    def test_add_with_pyobj(self):
        a = blaze.array(3) + 3
        self.assertEqual(dd_as_py(a._data), 6)
        a = 3 + blaze.array(4)
        self.assertEuqal(dd_as_py(a._data), 7)
        a = blaze.array([1, 2]) + 4
        self.assertEqual(dd_as_py(a._data), [5, 6])
        a = [1, 2] + blaze.array(5)
        self.assertEqual(dd_as_py(a._data), [6, 7])

    #FIXME:  Need to convert uint8 from dshape to ctypes
    #        in _get_ctypes of blaze_kernel.py
    def test_mixed(self):
        types1 = ['int8', 'int16', 'int32', 'int64']
        types2 = ['int16', 'int32', 'float32', 'float64']
        for ty1, ty2 in zip(types1, types2):
            a = blaze.array(range(1,6), dshape=ty1)
            b = blaze.array(range(5), dshape=ty2)
            c = (a+b)*(a-b)
            c = blaze.eval(c)
            result = [a*a - b*b for (a,b) in zip(range(1,6),range(5))]
            self.assertEqual(dd_as_py(c._data), result)

    def test_ragged(self):
        a = blaze.array([[1], [2, 3], [4, 5, 6]])
        b = blaze.array([[1, 2, 3], [4, 5], [6]])
        c = blaze.eval(a + b)
        self.assertEqual(dd_as_py(c._data),
                    [[2, 3, 4], [6, 8], [10, 11, 12]])
        c = blaze.eval(2 * a - b)
        self.assertEqual(dd_as_py(c._data),
                    [[1, 0, -1], [0, 1], [2, 4, 6]])

if __name__ == '__main__':
    unittest.main()
