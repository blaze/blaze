from __future__ import absolute_import, division, print_function

import unittest

from datashape import dshape
from dynd import nd, ndt
from blaze import array
from blaze.compute.ops.ufuncs import add, multiply


class TestGraph(unittest.TestCase):

    def test_graph(self):
        a = array(nd.range(10, dtype=ndt.int32))
        b = array(nd.range(10, dtype=ndt.float32))
        expr = add(a, multiply(a, b))
        graph, ctx = expr.expr
        self.assertEqual(len(ctx.params), 2)
        self.assertFalse(ctx.constraints)
        self.assertEqual(graph.dshape, dshape('10, float64'))

    def test_constant_folding(self):
        a = array(nd.range(10, dtype=ndt.int32))
        b = a + 1 + 1 + 1
        c = a + 3

        depth = lambda node: 1 if len(node.args) == 0 \
                               else 1 + depth(node.args[0])

        self.assertEqual(depth(b.expr[0]), depth(c.expr[0]))

if __name__ == '__main__':
    unittest.main()
