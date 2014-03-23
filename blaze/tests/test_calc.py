from __future__ import absolute_import, division, print_function

import unittest

import blaze
from blaze.datadescriptor import dd_as_py


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
        self.assertEqual(dd_as_py(a._data), 7)
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


class TestReduction(unittest.TestCase):
    def test_min_zerosize(self):
        # Empty min operations should raise, because it has no
        # reduction identity
        self.assertRaises(ValueError, blaze.eval, blaze.min([]))
        self.assertRaises(ValueError, blaze.eval, blaze.min([], keepdims=True))
        self.assertRaises(ValueError, blaze.eval, blaze.min([[], []]))
        self.assertRaises(ValueError, blaze.eval, blaze.min([[], []],
                                                            keepdims=True))
        self.assertRaises(ValueError, blaze.eval, blaze.min([[], []], axis=-1))
        self.assertRaises(ValueError, blaze.eval, blaze.min([[], []],
                                                            axis=-1,
                                                            keepdims=True))
        # However, if we're only reducing on a non-empty dimension, it's ok
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[], []],
                                                       axis=0))._data),
                         [])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[], []],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[]])

    def test_min(self):
        # Min element of scalar case is the element itself
        self.assertEqual(dd_as_py(blaze.eval(blaze.min(10))._data), 10)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min(-5.0))._data), -5.0)
        # One-dimensional size one
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([10]))._data), 10)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([-5.0]))._data), -5.0)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([-5.0],
                                                       axis=0))._data), -5.0)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([10],
                                                       keepdims=True))._data),
                         [10])
        # One dimensional
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([1, 2]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([2, 1]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([0, 1, 0]))._data), 0)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([0, 1, 0]))._data), 0)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([1, 0, 2]))._data), 0)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([2, 1, 0]))._data), 0)
        # Two dimensional, test with minimum at all possible positions
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[2, 1, 3],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[3, 2, 1],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [4, 1, 6]]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [4, 6, 1]]))._data), 1)
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [1, 6, 4]]))._data), 1)
        # Two dimensional, with axis= argument both positive and negative
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=0))._data),
                         [1, 2, 3])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=-2))._data),
                         [1, 2, 3])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=1))._data),
                         [1, 4])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=-1))._data),
                         [1, 4])
        # Two dimensional, with keepdims=True
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       keepdims=True))._data),
                         [[1]])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [5, 4, 6]],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[1, 2, 3]])
        self.assertEqual(dd_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=1,
                                                       keepdims=True))._data),
                         [[1], [2]])


if __name__ == '__main__':
    unittest.main()
