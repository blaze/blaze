# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import error
from blaze import dshape
from blaze.util import IdentityDict
from blaze.datashape import normalize_ellipses, numeric, normalize, simplify

class TestNormalization(unittest.TestCase):

    def test_normalize_ellipses1(self):
        ds1 = dshape('..., T')
        ds2 = dshape('X, Y, T')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'X, Y, T')
        self.assertEqual(str(res2), 'X, Y, T')

    def test_normalize_ellipses2(self):
        ds1 = dshape('A, ..., int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'A, Y, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses3(self):
        ds1 = dshape('..., A, int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'X, A, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses4(self):
        ds1 = dshape('..., A, B, int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'A, B, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses5(self):
        ds1 = dshape('..., A, B, int32')
        ds2 = dshape('..., X, Y, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), '..., A, B, int32')
        self.assertEqual(str(res2), '..., X, Y, float32')

    def test_normalize_ellipses_2_ellipses(self):
        ds1 = dshape('...,    A, int32')
        ds2 = dshape('X, ..., Y, Z, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'X, ..., Y, A, int32')
        self.assertEqual(str(res2), 'X, ..., Y, Z, float32')

    def test_normalize_ellipses_2_ellipses2(self):
        ds1 = dshape('A, ..., B, int32')
        ds2 = dshape('M, N, ..., S, T, float32')
        res1, res2 = normalize_ellipses(ds1, ds2)
        self.assertEqual(str(res1), 'A, N, ..., S, B, int32')
        self.assertEqual(str(res2), 'M, N, ..., S, T, float32')

    def test_normalize_ellipses_2_ellipses3_error(self):
        ds1 = dshape('A, ..., int32')
        ds2 = dshape('..., B, float32')
        self.assertRaises(error.BlazeTypeError, normalize_ellipses, ds1, ds2)

    def test_normalize_ellipses_2_ellipses4_error(self):
        ds1 = dshape('..., A, int32')
        ds2 = dshape('B, ..., float32')
        self.assertRaises(error.BlazeTypeError, normalize_ellipses, ds1, ds2)

    def test_normalize_ellipsis_context_error(self):
        ds1 = dshape('T, int32 -> T, T, int32')
        ds2 = dshape('A..., int32 -> A..., float32')
        [(x, y)], _ = normalize([(ds1, ds2)])
        self.assertEqual(str(y), '1, T, int32 -> T, T, float32')


class TestSimplification(unittest.TestCase):

    def test_simplify_implements(self):
        ds = dshape('10, A : numeric')
        self.assertEqual(str(ds), '10, A : numeric')

        solution = IdentityDict()
        ds = simplify(ds, solution)

        self.assertEqual(str(ds), '10, A')
        A = ds.parameters[-1]
        [x] = solution[A]
        self.assertEqual(set(x), set(numeric))


if __name__ == '__main__':
    # TestSimplification('test_simplify_implements').debug()
    unittest.main()