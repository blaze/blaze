# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze.kernel import kernel
from blaze import dshape, array

# f

@kernel('X, Y, float32 -> X, Y, float32 -> X, Y, float32')
def f(a, b):
    return a

@kernel('X, Y, complex64 -> X, Y, complex64 -> X, Y, complex64')
def f(a, b):
    return a

@kernel('X, Y, complex128 -> X, Y, complex128 -> X, Y, complex128')
def f(a, b):
    return a

# g

@kernel('X, Y, float32 -> X, Y, float32 -> X, int32')
def g(a, b):
    return a

@kernel('X, Y, float32 -> ..., float32 -> X, int32')
def g(a, b):
    return a

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestBlazeKernel(unittest.TestCase):

    def test_kernel(self):
        A = array([8, 9], dshape('2, int32'))
        res = f(A, A)
        self.assertEqual(str(res.dshape), '1, 2, float32')
        self.assertEqual(len(res.expr), 2)
        graph, ctx = res.expr
        self.assertEqual(len(graph.args), 2)
        self.assertEqual(len(ctx.constraints), 0)
        self.assertEqual(len(ctx.params), 1)
        # res.view()


if __name__ == '__main__':
    # TestBlazeKernel('test_kernel').debug()
    unittest.main()