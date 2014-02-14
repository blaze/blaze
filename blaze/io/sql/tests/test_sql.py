from __future__ import print_function, division, absolute_import

import unittest
from datashape import dshape

from blaze import add, multiply, eval, py2help
from blaze.io.sql import from_table, db
from blaze.io.sql.ops import index
from blaze.io.sql.tests.testutils import create_sqlite_table

import numpy as np

class TestSQL(unittest.TestCase):

    def setUp(self):
        self.conn = create_sqlite_table()

        self.col_i = from_table('testtable', 'i',
                                dshape('3, int32'),
                                self.conn)
        self.col_msg = from_table('testtable', 'msg',
                                  dshape('3, string'),
                                  self.conn)
        self.col_price = from_table('testtable', 'price',
                                    dshape('3, float64'),
                                    self.conn)

    @py2help.skipIf(db is None, 'pyodbc is not installed')
    def test_query(self):
        expr = add(self.col_i, self.col_i)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [8, 16, 32])

    @py2help.skipIf(db is None, 'pyodbc is not installed')
    def test_query_scalar(self):
        expr = add(self.col_i, 2)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [6, 10, 18])

    @py2help.skipIf(db is None, 'pyodbc is not installed')
    def test_query_where(self):
        expr = index(self.col_i + self.col_i, self.col_i > 5)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [16, 32])


if __name__ == '__main__':
    #TestSQL('test_query_where').debug()
    unittest.main()
