from __future__ import print_function, division, absolute_import

import unittest
from datashape import dshape

from blaze import add, multiply, eval, py2help
from blaze.io.sql import sql_table, sql_column, db
from blaze.io.sql.ops import index
from blaze.io.sql.tests.testutils import create_sqlite_table


class TestSQL(unittest.TestCase):

    def setUp(self):
        self.conn = create_sqlite_table()

        self.table = sql_table(
            'testtable',
            ['i', 'msg', 'price'],
            [dshape('int32'), dshape('string'), dshape('float64')],
            self.conn)

        self.col_i = sql_column('testtable', 'i',
                                dshape('3, int32'),
                                self.conn)
        self.col_msg = sql_column('testtable', 'msg',
                                  dshape('3, string'),
                                  self.conn)
        self.col_price = sql_column('testtable', 'price',
                                    dshape('3, float64'),
                                    self.conn)


class TestSQLColumns(TestSQL):

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


class TestSQLTable(TestSQL):

    @py2help.skipIf(db is None, 'pyodbc is not installed')
    def test_query_where(self):
        expr = index(self.table, self.col_i > 5)
        result = eval(expr)
        row1, row2 = result
        self.assertEqual((int(row1[0]), str(row1[1]), float(row1[2])),
                         (8, "world", 4.2))
        self.assertEqual((int(row2[0]), str(row2[1]), float(row2[2])),
                         (16, "!", 8.4))


if __name__ == '__main__':
    # TestSQLTable('test_query').debug()
    unittest.main()
