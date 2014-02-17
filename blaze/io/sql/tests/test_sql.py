from __future__ import print_function, division, absolute_import

import unittest
from datashape import dshape, bool_

import blaze
from blaze import add, multiply, eval, py2help
from blaze.io.sql import sql_table, sql_column, db
from blaze.io.sql import ops
from blaze.io.sql.tests.testutils import create_sqlite_table, data

import numpy as np

skipif = py2help.skipIf(db is None, 'pyodbc is not installed')

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


class TestSQLOps(TestSQL):

    ## ufuncs

    @skipif
    def test_add_scalar(self):
        expr = self.col_i + 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [6, 10, 18])

    @skipif
    def test_sub_scalar(self):
        expr = self.col_i - 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 6, 14])

    @skipif
    def test_mul_scalar(self):
        expr = self.col_i * 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [8, 16, 32])

    @skipif
    def test_floordiv_scalar(self):
        expr = self.col_i // 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 4, 8])

    @skipif
    def test_truediv_scalar(self):
        expr = self.col_i / 2
        result = eval(expr)
        self.assertEqual([float(x) for x in result], [2., 4., 8.])

    @skipif
    def test_mod_scalar(self):
        expr = self.col_i % 3
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [1, 2, 1])

    @skipif
    def test_neg_scalar(self):
        expr = -self.col_i
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [-4, -8, -16])

    ## compare

    @skipif
    def test_eq_scalar(self):
        expr = self.col_i == 8
        result = eval(expr)
        self.assertEqual(result.dshape.measure, bool_)
        self.assertEqual([bool(x) for x in result], [False, True, False])

    @skipif
    def test_ne_scalar(self):
        expr = self.col_i != 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipif
    def test_lt_scalar(self):
        expr = self.col_i < 5
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, False])


    @skipif
    def test_le_scalar(self):
        expr = self.col_i <= 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, True, False])


    @skipif
    def test_gt_scalar(self):
        expr = self.col_i > 9
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, False, True])


    @skipif
    def test_ge_scalar(self):
        expr = self.col_i >= 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, True])

    ## logical

    @skipif
    def test_and(self):
        expr = blaze.logical_and(5 < self.col_i, self.col_i < 10)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, False])

    @skipif
    def test_or(self):
        expr = blaze.logical_or(self.col_i < 5, self.col_i > 10)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipif
    def test_xor(self):
        expr = blaze.logical_xor(self.col_i < 9, self.col_i > 6)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipif
    def test_not(self):
        expr = blaze.logical_not(self.col_i < 5)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, True])


class TestSQLUFuncExpressions(TestSQL):

    @skipif
    def test_select_expr(self):
        expr = ((ops.max(self.col_price) / ops.min(self.col_price)) *
                (self.col_i + 2) * 3.1 -
                ops.avg(self.col_i))
        result = eval(expr)

        test_data = np.array(data, dtype=[('i', np.int32),
                                          ('msg', '|S5'),
                                          ('price', np.float64)])
        i, price = test_data['i'], test_data['price']
        np_result = ((np.max(price) / np.min(price)) *
                     (i + 2) * 3.1 -
                     np.average(i) / np.max(price))

        self.assertEqual([float(x) for x in result],
                         [float(x) for x in np_result])


class TestSQLDataTypes(TestSQL):

    @skipif
    def test_int(self):
        expr = self.col_i // 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 4, 8])


class TestSQLColumns(TestSQL):

    @skipif
    def test_query(self):
        expr = add(self.col_i, self.col_i)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [8, 16, 32])

    @skipif
    def test_query_scalar(self):
        expr = add(self.col_i, 2)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [6, 10, 18])

    @skipif
    def test_query_where(self):
        expr = ops.index(self.col_i + self.col_i, self.col_i > 5)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [16, 32])


class TestSQLTable(TestSQL):

    @skipif
    def test_query_where(self):
        expr = ops.index(self.table, self.col_i > 5)
        result = eval(expr)
        row1, row2 = result
        self.assertEqual((int(row1[0]), str(row1[1]), float(row1[2])),
                         (8, "world", 4.2))
        self.assertEqual((int(row2[0]), str(row2[1]), float(row2[2])),
                         (16, "!", 8.4))


if __name__ == '__main__':
    # TestSQLTable('test_query').debug()
    # TestSQLUFuncExpressions('test_select_expr').debug()
    unittest.main()
