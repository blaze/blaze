from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
from datashape import coretypes as T
from blaze.compute.function import blaze_func, kernel


class TestBlazeFunction(unittest.TestCase):

    def test_define_dynamic(self):
        # Define an element-wise blaze function
        f = blaze_func("test_func", dshape("(A... * T, A... * T) -> A... * T"), elementwise=True)

        # Define implementation of element-wise blaze function
        # use implementation category 'funky'
        signature1 = T.Function(*[dshape("float64")] * 3)
        kernel1 = lambda a, b: a * b
        kernel(f, 'funky', kernel1, signature1)

        signature2 = T.Function(*[dshape("Axes... * float64")] * 3)
        kernel2 = lambda a, b: a * b
        kernel(f, 'funky', kernel2, signature2)

        # See that we can find the right 'funky' implementation
        overload = f.best_match('funky',
                                T.Tuple([dshape("float32"), dshape("float64")]))
        self.assertEqual(overload.resolved_sig, signature1)
        self.assertEqual(overload.func, kernel1)

        overload = f.best_match('funky', T.Tuple([dshape("10 * 10 * float32"),
                                                  dshape("10 * 10 * float64")]))
        self.assertEqual(overload.resolved_sig,
                         dshape("(10 * 10 * float64, 10 * 10 * float64) -> 10 * 10 * float64")[0])
        self.assertEqual(overload.func, kernel2)


if __name__ == '__main__':
    unittest.main()
