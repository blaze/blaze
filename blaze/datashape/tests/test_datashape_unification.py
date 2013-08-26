import re
import unittest

import blaze
from blaze import error
from blaze import datashape, dshape
from blaze.datashape import (unify, coretypes as T, DataShape,
                             normalize_simple as normalize)


class TestNormalization(unittest.TestCase):

    def test_normalize_ellipses1(self):
        ds1 = dshape('..., T')
        ds2 = dshape('X, Y, T')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'X, Y, T')
        self.assertEqual(str(res2), 'X, Y, T')

    def test_normalize_ellipses2(self):
        ds1 = dshape('A, ..., int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'A, Y, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses3(self):
        ds1 = dshape('..., A, int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'X, A, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses4(self):
        ds1 = dshape('..., A, B, int32')
        ds2 = dshape('X, Y, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'A, B, int32')
        self.assertEqual(str(res2), 'X, Y, float32')

    def test_normalize_ellipses5(self):
        ds1 = dshape('..., A, B, int32')
        ds2 = dshape('..., X, Y, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), '..., A, B, int32')
        self.assertEqual(str(res2), '..., X, Y, float32')

    def test_normalize_ellipses_2_ellipses(self):
        ds1 = dshape('...,    A, int32')
        ds2 = dshape('X, ..., Y, Z, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'X, ..., Y, A, int32')
        self.assertEqual(str(res2), 'X, ..., Y, Z, float32')

    def test_normalize_ellipses_2_ellipses2(self):
        ds1 = dshape('A, ..., B, int32')
        ds2 = dshape('M, N, ..., S, T, float32')
        res1, res2 = normalize(ds1, ds2)
        self.assertEqual(str(res1), 'A, N, ..., S, B, int32')
        self.assertEqual(str(res2), 'M, N, ..., S, T, float32')

    def test_normalize_ellipses_2_ellipses3_error(self):
        ds1 = dshape('A, ..., int32')
        ds2 = dshape('..., B, float32')
        self.assertRaises(error.BlazeTypeError, normalize, ds1, ds2)

    def test_normalize_ellipses_2_ellipses4_error(self):
        ds1 = dshape('..., A, int32')
        ds2 = dshape('B, ..., float32')
        self.assertRaises(error.BlazeTypeError, normalize, ds1, ds2)


def symsub(ds, S):
    """Substitute type variables by name"""
    return DataShape([S.get(x.symbol, x) if isinstance(x, T.TypeVar) else x
                          for x in ds.parameters])


class TestUnification(unittest.TestCase):

    def test_unify_datashape_promotion(self):
        d1 = dshape('10, T1, int32')
        d2 = dshape('T2, T2, float32')
        [result], constraints = unify([(d1, d2)], [True])
        self.assertEqual(result, dshape('10, 10, float32'))

    def test_unify_datashape_promotion2(self):
        A, B = T.TypeVar('A'), T.TypeVar('B')
        X, Y, Z = T.TypeVar('X'), T.TypeVar('Y'), T.TypeVar('Z')
        S = dict((typevar.symbol, typevar) for typevar in (A, B, X, Y, Z))

        # LHS
        d1 = dshape('A, B, int32')
        d2 = dshape('B, 10, float32')

        # RHS
        d3 = dshape('X, Y, int16')
        d4 = dshape('X, X, Z')

        # Create proper equation
        d1, d2, d3, d4 = [symsub(ds, S) for ds in (d1, d2, d3, d4)]
        constraints = [(d1, d3), (d2, d4)]

        # What we know from the above equations is:
        #   1) A coerces to X
        #   2) B coerces to Y
        #   3) 10 coerces to X
        #
        # From this we determine that X must be Fixed(10). We must retain
        # type variable B for Y, since we can only say that B must unify with
        # Fixed(10), but not whether it is actually Fixed(10) (it may also be
        # Fixed(1))

        [arg1, arg2], remaining_constraints = unify(constraints, [True, True])
        self.assertEqual(arg1, dshape('10, B, int16'))
        self.assertEqual(arg2, dshape('10, 10, float32'))

    def test_unify_datashape_error(self):
        d1 = dshape('10, 11, int32')
        d2 = dshape('T2, T2, int32')
        self.assertRaises(error.UnificationError, unify, [(d1, d2)], [True])


if __name__ == '__main__':
    # TestUnification('test_unify_datashape_promotion').debug()
    # TestNormalization('test_normalize_ellipses5').debug()
    unittest.main()