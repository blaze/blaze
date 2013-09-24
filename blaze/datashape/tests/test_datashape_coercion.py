# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import error
from blaze.tests import common
from blaze.datashape import unify, dshapes, coerce, normalize

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestCoercion(common.BTestCase):

    def test_coerce_ctype(self):
        a, b = dshapes('float32', 'float32')
        self.assertEqual(coerce(a, b), 0)

    def test_coerce_numeric(self):
        a, b = dshapes('float32', 'float64')
        self.assertEqual(coerce(a, b), 1)

    def test_coercion_transitivity(self):
        a, b, c = dshapes('int8', 'complex128', 'float64')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_typevars(self):
        a, b, c = dshapes('10, 11, float32', 'X, Y, float64', '10, Y, float64')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_constrained_typevars(self):
        a, b, c = dshapes('10, 10, float32', 'X, Y, float64', 'X, X, float64')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_broadcasting(self):
        a, b, c = dshapes('10, 10, float32', '10, Y, Z, float64', 'X, Y, float64')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_broadcasting2(self):
        a, b, c = dshapes('10, 10, float32', '1, 10, 10, float32', '10, 10, float32')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_broadcasting3(self):
        a, b, c = dshapes('10, 10, float32', '10, 10, 10, float32', '1, 10, 10, float32')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_traits(self):
        a, b, c = dshapes('10, 10, float32', '10, X, A : floating', '10, X, float32')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_dst_ellipsis(self):
        a, b, c = dshapes('10, 10, float32', 'X, ..., float64', 'X, Y, float64')
        self.assertGreater(coerce(a, b), coerce(a, c))

    def test_coerce_src_ellipsis(self):
        a, b, c = dshapes('10, ..., float32', 'X, Y, float64', 'X, ..., float64')
        self.assertGreater(coerce(a, b), coerce(a, c))


class TestCoercionErrors(unittest.TestCase):

    def test_downcast(self):
        a, b = dshapes('float32', 'int32')
        self.assertRaises(error.CoercionError, coerce, a, b)


if __name__ == '__main__':
    # TestCoercion('test_coerce_src_ellipsis').debug()
    unittest.main()