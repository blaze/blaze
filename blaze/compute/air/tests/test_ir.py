from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
from blaze.compute.air.tests.utils import make_graph


class TestIR(unittest.TestCase):

    def test_ir(self):
        f, values, graph = make_graph()

        # Structure
        self.assertEqual(len(f.blocks), 1)
        self.assertTrue(f.startblock.is_terminated())

        # Types
        got      = [op.type for op in f.ops][:-1]
        expected = [dshape("10, float64"), dshape("10, complex[float64]")]
        self.assertEqual(got, expected)

        # function 10, complex[float64] expr0(10, int32 %e0, 10, float64 %e1, 10, complex[float64] %e2) {
        # entry:
        #     %0 = (10, float64) kernel(%const(Bytes, blaze.ops.ufuncs.add), %e0, %e1)
        #     %1 = (10, complex[float64]) kernel(%const(Bytes, blaze.ops.ufuncs.mul), %0, %e2)
        #     %2 = (Void) ret(%1)
        #
        # }


if __name__ == '__main__':
    # TestIR('test_ir').debug()
    unittest.main()
