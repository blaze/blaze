# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import error
from blaze.datashape import unify_simple, promote, coerce, dshapes, coretypes as T

#------------------------------------------------------------------------
# Test data
#------------------------------------------------------------------------

Complex = T.TypeConstructor('Complex', 1, [{'coercible': True}])
t1 = Complex(T.int64)
t2 = Complex(T.int64)
t3 = Complex(T.int32)

RigidComplex = T.TypeConstructor('Complex', 1, [{'coercible': False}])
rt1 = RigidComplex(T.int64)
rt2 = RigidComplex(T.int64)
rt3 = RigidComplex(T.int32)

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestTypeConstructors(unittest.TestCase):

    def test_equality(self):
        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_unification_concrete(self):
        self.assertEqual(unify_simple(t1, t2), t1)

    def test_unification_typevar(self):
        tvar = Complex(T.TypeVar('A'))
        self.assertEqual(unify_simple(t1, tvar), t1)

    def test_promotion(self):
        self.assertEqual(promote(t1, t2), t1)
        self.assertEqual(promote(t1, t3), t1)
        self.assertEqual(promote(t3, t2), t1)
        self.assertEqual(promote(rt1, rt2), rt1)

    def test_coercion(self):
        self.assertEqual(coerce(t1, t2), 0)
        self.assertGreater(coerce(t3, t2), 0)
        self.assertEqual(coerce(rt1, rt2), 0)


class TestErrors(unittest.TestCase):

    def test_promotion_error(self):
        self.assertRaises(error.UnificationError, promote, rt1, rt3)

if __name__ == '__main__':
    unittest.main()