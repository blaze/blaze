# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from .utils import make_graph

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestIR(unittest.TestCase):

    def test_ir(self):
        f, values, graph = make_graph()

        self.assertEqual(len(f.blocks), 1)
        self.assertTrue(f.startblock.is_terminated())

        # function 10, cfloat64 expr0(10, float64 %e0, 10, int32 %e1, 10, cfloat64 %e2) {
        # entry:
        #     %0 = (Opaque) kernel(%const(Bytes, blaze.ops.ufuncs.add), %e1, %e0)
        #     %1 = (Opaque) kernel(%const(Bytes, blaze.ops.ufuncs.mul), %0, %e2)
        #     %2 = (Void) ret(%1)
        #
        # }

    def test_conversions(self):
        f, values, graph = make_graph()


if __name__ == '__main__':
    # TestIR('test_ir').debug()
    unittest.main()