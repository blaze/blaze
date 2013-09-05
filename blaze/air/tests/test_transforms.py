# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze.air import explicit_coercions
from blaze.air.tests.utils import make_graph

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

class TestCoercions(unittest.TestCase):

    def test_coercions(self):
        f, values, graph = make_graph()
        explicit_coercions(f)

        print(f)

        # function 10, cfloat64 expr0(10, float64 %e0, 10, int32 %e1, 10, cfloat64 %e2) {
        # entry:
        #     %0 = (Opaque) kernel(%const(Bytes, blaze.ops.ufuncs.add), %e1, %e0)
        #     %1 = (Opaque) kernel(%const(Bytes, blaze.ops.ufuncs.mul), %0, %e2)
        #     %2 = (Void) ret(%1)
        #
        # }



if __name__ == '__main__':
    TestCoercions('test_coercions').debug()
    # unittest.main()