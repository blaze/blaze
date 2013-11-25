# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest

from blaze import dshape, add, mul, eval
from blaze.scidb import connect, empty, zeros, ones
from blaze.scidb.tests.mock import MockedConn

ds = dshape('10, 10, int32')

class TestSciDB(unittest.TestCase):

    def setUp(self):
        self.conn = MockedConn()

    def test_query(self):
        a = zeros(ds, self.conn)
        b = ones(ds, self.conn)

        expr = add(a, mul(a, b))

        graph, ctx = expr.expr
        self.assertEqual(graph.dshape, dshape('10, 10, int32'))

        result = eval(expr)

if __name__ == '__main__':
    #unittest.main()
    TestSciDB('test_query').debug()