import unittest

from blaze import dshape, array
from blaze.ops.ufuncs import add, mul

import numpy as np

class TestGraph(unittest.TestCase):

    def test_graph(self):
        a = array(np.arange(10), dshape=dshape('10, int32'))
        b = array(np.arange(10), dshape=dshape('10, float32'))
        expr = add(a, mul(a, b))
        graph, ctx = expr.expr
        self.assertEqual(len(ctx.params), 2)
        self.assertFalse(ctx.constraints)
        self.assertEqual(graph.dshape, dshape('10, float64'))


if __name__ == '__main__':
    unittest.main()