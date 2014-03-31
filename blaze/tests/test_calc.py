from __future__ import absolute_import, division, print_function

import unittest

import blaze
from blaze.datadescriptor import ddesc_as_py


class TestBasic(unittest.TestCase):

    def test_add(self):
        types = ['int8', 'int16', 'int32', 'int64']
        for type_ in types:
            a = blaze.array(range(3), dshape=type_)
            c = blaze.eval(a+a)
            self.assertEqual(ddesc_as_py(c._data), [0, 2, 4])
            c = blaze.eval(((a+a)*a))
            self.assertEqual(ddesc_as_py(c._data), [0, 2, 8])

    def test_add_with_pyobj(self):
        a = blaze.array(3) + 3
        self.assertEqual(ddesc_as_py(a._data), 6)
        a = 3 + blaze.array(4)
        self.assertEqual(ddesc_as_py(a._data), 7)
        a = blaze.array([1, 2]) + 4
        self.assertEqual(ddesc_as_py(a._data), [5, 6])
        a = [1, 2] + blaze.array(5)
        self.assertEqual(ddesc_as_py(a._data), [6, 7])

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
            self.assertEqual(ddesc_as_py(c._data), result)

    def test_ragged(self):
        a = blaze.array([[1], [2, 3], [4, 5, 6]])
        b = blaze.array([[1, 2, 3], [4, 5], [6]])
        c = blaze.eval(a + b)
        self.assertEqual(ddesc_as_py(c._data),
                    [[2, 3, 4], [6, 8], [10, 11, 12]])
        c = blaze.eval(2 * a - b)
        self.assertEqual(ddesc_as_py(c._data),
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
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[], []],
                                                       axis=0))._data),
                         [])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[], []],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[]])

    def test_min(self):
        # Min element of scalar case is the element itself
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min(10))._data), 10)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min(-5.0))._data), -5.0)
        # One-dimensional size one
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([10]))._data), 10)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([-5.0]))._data), -5.0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([-5.0],
                                                       axis=0))._data), -5.0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([10],
                                                       keepdims=True))._data),
                         [10])
        # One dimensional
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([1, 2]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([2, 1]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([0, 1, 0]))._data), 0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([0, 1, 0]))._data), 0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([1, 0, 2]))._data), 0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([2, 1, 0]))._data), 0)
        # Two dimensional, test with minimum at all possible positions
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[2, 1, 3],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[3, 2, 1],
                                                        [4, 5, 6]]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [4, 1, 6]]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [4, 6, 1]]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[3, 2, 5],
                                                        [1, 6, 4]]))._data), 1)
        # Two dimensional, with axis= argument both positive and negative
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=0))._data),
                         [1, 2, 3])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=-2))._data),
                         [1, 2, 3])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=1))._data),
                         [1, 4])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=-1))._data),
                         [1, 4])
        # Two dimensional, with keepdims=True
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       keepdims=True))._data),
                         [[1]])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 2, 3],
                                                        [5, 4, 6]],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[1, 2, 3]])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.min([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=1,
                                                       keepdims=True))._data),
                         [[1], [2]])

    def test_sum_zerosize(self):
        # Empty sum operations should produce 0, the reduction identity
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([]))._data), 0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([],
                                                       keepdims=True))._data),
                         [0])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []]))._data), 0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []],
                                                       keepdims=True))._data),
                         [[0]])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []],
                                                       axis=-1))._data),
                         [0, 0])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []],
                                                            axis=-1,
                                                            keepdims=True))._data),
                         [[0], [0]])
        # If we're only reducing on a non-empty dimension, we might still
        # end up with zero-sized outputs
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []],
                                                       axis=0))._data),
                         [])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[], []],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[]])

    def test_sum(self):
        # Sum of scalar case is the element itself
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum(10))._data), 10)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum(-5.0))._data), -5.0)
        # One-dimensional size one
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([10]))._data), 10)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([-5.0]))._data), -5.0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([-5.0],
                                                       axis=0))._data), -5.0)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([10],
                                                       keepdims=True))._data),
                         [10])
        # One dimensional
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([1, 2]))._data), 3)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([0, 1, 2]))._data), 3)
        # Two dimensional
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 2, 3],
                                                        [4, 5, 6]]))._data), 21)
        # Two dimensional, with axis= argument both positive and negative
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=0))._data),
                         [5, 7, 9])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=-2))._data),
                         [5, 7, 9])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=1))._data),
                         [6, 15])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       axis=-1))._data),
                         [6, 15])
        # Two dimensional, with keepdims=True
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 2, 3],
                                                        [4, 5, 6]],
                                                       keepdims=True))._data),
                         [[21]])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 2, 3],
                                                        [5, 4, 6]],
                                                       axis=0,
                                                       keepdims=True))._data),
                         [[6, 6, 9]])
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.sum([[1, 5, 3],
                                                        [4, 2, 6]],
                                                       axis=1,
                                                       keepdims=True))._data),
                         [[9], [12]])

    def test_all(self):
        # Sanity check of reduction op
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.all(True))._data), True)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.all(False))._data), False)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.all(blaze.array([], dshape='0 * bool')))._data), True)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.all([False, True]))._data),
                         False)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.all([True, True]))._data),
                         True)

    def test_any(self):
        # Sanity check of reduction op
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.any(True))._data), True)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.any(False))._data), False)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.any(blaze.array([], dshape='0 * bool')))._data), False)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.any([False, True]))._data),
                         True)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.any([False, False]))._data),
                         False)

    def test_max(self):
        # Sanity check of reduction op
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.max(5))._data), 5)
        self.assertRaises(ValueError, blaze.eval, blaze.max([]))
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.max([3, -2]))._data),
                         3)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.max([1.5, 2.0]))._data),
                         2.0)

    def test_product(self):
        # Sanity check of reduction op
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.product(5))._data), 5)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.product([]))._data), 1)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.product([3, -2]))._data),
                         -6)
        self.assertEqual(ddesc_as_py(blaze.eval(blaze.product([1.5, 2.0]))._data),
                         3.0)



if __name__ == '__main__':
    unittest.main()
