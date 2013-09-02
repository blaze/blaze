# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import error
from blaze.overloading import best_match, overload
from blaze import dshape
from blaze.datashape import unify, unify_simple, dshapes

# f

@overload('X, Y, float32 -> X, Y, float32 -> X, Y, float32')
def f(a, b):
    return a

@overload('X, Y, complex64 -> X, Y, complex64 -> X, Y, complex64')
def f(a, b):
    return a

@overload('X, Y, complex128 -> X, Y, complex128 -> X, Y, complex128')
def f(a, b):
    return a

# g

@overload('X, Y, float32 -> X, Y, float32 -> X, int32')
def g(a, b):
    return a

@overload('X, Y, float32 -> ..., float32 -> X, int32')
def g(a, b):
    return a

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestOverloading(unittest.TestCase):

    def test_best_match(self):
        d1 = dshape('10, T1, int32')
        d2 = dshape('T2, T2, float32')
        dst_sig, sig, func = best_match(f, [d1, d2])
        self.assertEqual(str(sig),
                         'X, Y, float32 -> X, Y, float32 -> X, Y, float32')

        input = dshape('1, 1, float32 -> 1, 1, float32 -> R')
        self.assertEqual(str(unify_simple(input, dst_sig)),
                         '10, 1, float32 -> 10, 1, float32 -> 10, 1, float32')

    def test_best_match_broadcasting(self):
        d1 = dshape('10, complex64')
        d2 = dshape('10, float32')
        dst_sig, sig, func = best_match(f, [d1, d2])
        self.assertEqual(str(sig),
                         'X, Y, complex64 -> X, Y, complex64 -> X, Y, complex64')
        self.assertEqual(str(dst_sig),
                         '1, 10, complex64 -> 1, 10, complex64 -> 1, 10, complex64')

    def test_best_match_ellipses(self):
        d1 = dshape('10, T1, int32')
        d2 = dshape('..., float32')
        dst_sig, sig, func = best_match(g, [d1, d2])
        self.assertEqual(str(sig), 'X, Y, float32 -> ..., float32 -> X, int32')
        self.assertEqual(str(dst_sig),
                         '10, T1, float32 -> ..., float32 -> 10, int32')


if __name__ == '__main__':
    # TestOverloading('test_best_match_ellipses').debug()
    unittest.main()