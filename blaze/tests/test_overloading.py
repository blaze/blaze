from __future__ import print_function, division, absolute_import

import unittest

from datashape import dshape, unify_simple

from blaze.compute.overloading import best_match, overload
from blaze import py2help


#f

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


class TestOverloading(unittest.TestCase):

    def test_best_match(self):
        d1 = dshape('10, T1, int32')
        d2 = dshape('T2, T2, float32')
        match = best_match(f, [d1, d2])
        self.assertEqual(str(match.sig),
                         'X, Y, float32 -> X, Y, float32 -> X, Y, float32')

        input = dshape('S, 1, float32 -> T, 1, float32 -> R')
        self.assertEqual(str(unify_simple(input, match.resolved_sig)),
                         '10, 1, float32 -> 10, 1, float32 -> 10, 1, float32')

    @py2help.skip
    def test_best_match_broadcasting(self):
        d1 = dshape('10, complex64')
        d2 = dshape('10, float32')
        match = best_match(f, [d1, d2])
        self.assertEqual(str(match.sig),
                         'X, Y, complex[float32] -> X, Y, complex[float32] -> X, Y, complex[float32]')
        self.assertEqual(str(match.resolved_sig),
                         '1, 10, complex[float32] -> 1, 10, complex[float32] -> 1, 10, complex[float32]')

    def test_best_match_ellipses(self):
        d1 = dshape('10, T1, int32')
        d2 = dshape('..., float32')
        match = best_match(g, [d1, d2])
        self.assertEqual(str(match.sig), 'X, Y, float32 -> ..., float32 -> X, int32')
        self.assertEqual(str(match.resolved_sig),
                         '10, T1, float32 -> ..., float32 -> 10, int32')


if __name__ == '__main__':
    #TestOverloading('test_best_match_broadcasting').debug()
    unittest.main()
