from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
from datashape import coretypes as T
from blaze.compute.function import blaze_func, blaze_func_from_nargs, kernel


class TestBlazeFunction(unittest.TestCase):

    def test_define_dynamic(self):
        # Define an element-wise blaze function
        f = blaze_func("test_func", dshape("a -> a -> a"), elementwise=True)

        # Define implementation of element-wise blaze function
        # use implementation category 'funky'
        signature1 = T.Function(*[dshape("float64")] * 3)
        kernel1 = lambda a, b: a * b
        kernel(f, 'funky', kernel1, signature1)

        signature2 = T.Function(*[dshape("axes..., float64")] * 3)
        kernel2 = lambda a, b: a * b
        kernel(f, 'funky', kernel2, signature2)

        # See that we can find the right 'funky' implementation
        overload = f.best_match('funky', [dshape("float32"), dshape("float64")])
        self.assertEqual(overload.resolved_sig, signature1)
        self.assertEqual(overload.func, kernel1)

        overload = f.best_match('funky', [dshape("10, 10, float32"),
                                          dshape("10, 10, float64")])
        self.assertEqual(overload.resolved_sig,
                         dshape("10, 10, float64 -> 10, 10, float64 -> 10, 10, float64"))
        self.assertEqual(overload.func, kernel2)


    def test_define_dynamic_nargs(self):
        # Define an element-wise blaze function
        f = blaze_func_from_nargs("test_func2", 2)

        # Define implementation of element-wise blaze function
        # use implementation category 'funky'
        signature = T.Function(*[dshape("float64")] * 3)
        kernel(f, 'funky', lambda a, b: a * b, signature)

        # See that we can find the right 'funky' implementation
        overload = f.best_match('funky', [dshape("float32"), dshape("float64")])
        self.assertEqual(overload.resolved_sig, signature)


if __name__ == '__main__':
    unittest.main()
