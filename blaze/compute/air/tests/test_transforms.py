# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from pykit.ir import opcodes
from blaze import dshape
from blaze.compute.air import explicit_coercions
from blaze.compute.air.tests.utils import make_graph

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestCoercions(unittest.TestCase):

    def test_coercions(self):
        f, values, graph = make_graph()
        explicit_coercions(f)
        ops = [(op.opcode, op.type) for op in f.ops][:-1]
        expected = [('convert', dshape("10, float64")),
                    ('kernel', dshape("10, float64")),
                    ('convert', dshape("10, cfloat64")),
                    ('kernel', dshape("10, cfloat64"))]
        self.assertEqual(ops, expected)

        # function 10, cfloat64 expr0(10, float64 %e0, 10, int32 %e1, 10, cfloat64 %e2) {
        # entry:
        #     %3 = (10, float64) convert(%e1)
        #     %0 = (10, float64) kernel(%const(Bytes, blaze.ops.ufuncs.add), %3, %e0)
        #     %4 = (10, cfloat64) convert(%0)
        #     %1 = (10, cfloat64) kernel(%const(Bytes, blaze.ops.ufuncs.mul), %4, %e2)
        #     %2 = (Void) ret(%1)
        #
        # }



if __name__ == '__main__':
    unittest.main()