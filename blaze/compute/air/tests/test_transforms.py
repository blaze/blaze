from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
from blaze.compute.air import explicit_coercions
from blaze.compute.air.tests.utils import make_graph


class TestCoercions(unittest.TestCase):

    def test_coercions(self):
        f, values, graph = make_graph()
        explicit_coercions(f)
        ops = [(op.opcode, op.type) for op in f.ops][:-1]
        expected = [('convert', dshape("10, float64")),
                    ('kernel', dshape("10, float64")),
                    ('convert', dshape("10, complex[float64]")),
                    ('kernel', dshape("10, complex[float64]"))]
        self.assertEqual(ops, expected)

        # function 10, complex[float64] expr0(10, float64 %e0, 10, int32 %e1, 10, complex[float64] %e2) {
        # entry:
        #     %3 = (10, float64) convert(%e1)
        #     %0 = (10, float64) kernel(%const(Bytes, blaze.ops.ufuncs.add), %3, %e0)
        #     %4 = (10, complex[float64]) convert(%0)
        #     %1 = (10, complex[float64]) kernel(%const(Bytes, blaze.ops.ufuncs.mul), %4, %e2)
        #     %2 = (Void) ret(%1)
        #
        # }


if __name__ == '__main__':
    unittest.main()
