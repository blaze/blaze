import re
import unittest

import blaze
from blaze import error
from blaze import datashape, dshape
from blaze.datashape import unify, coretypes as T, DataShape

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


# TestUnification('test_unify_datashape_promotion').debug()
# TestUnification('test_unify_datashape_promotion2').debug()
# TestUnification('test_unify_datashape_error').debug()
