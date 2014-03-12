from __future__ import print_function, division, absolute_import

import unittest

from nose.plugins.skip import SkipTest
import numpy as np

import blaze
from datashape import dshape, bool_

from blaze import add, multiply, eval
from blaze.io.sql import sql_table, sql_column, db
from blaze.io.sql import ops
from blaze.io.sql.tests.testutils import create_sqlite_table, data
from blaze.py2help import skip, skipIf


class TestSQL(unittest.TestCase):

    def setUp(self):
        self.conn = create_sqlite_table()

        self.table = sql_table(
            'testtable',
            ['i', 'msg', 'price'],
            [dshape('int32'), dshape('string'), dshape('float64')],
            self.conn)

        self.col_i = sql_column('testtable', 'i',
                                dshape('3 * int32'),
                                self.conn)
        self.col_msg = sql_column('testtable', 'msg',
                                  dshape('3 * string'),
                                  self.conn)
        self.col_price = sql_column('testtable', 'price',
                                    dshape('3 * float64'),
                                    self.conn)

        test_data = np.array(data, dtype=[('i', np.int32),
                                          ('msg', '|S5'),
                                          ('price', np.float64)])
        self.np_i = test_data['i']
        self.np_msg = test_data['msg']
        self.np_price = test_data['price']


class TestSQLOps(TestSQL):

    ## ufuncs

    @skipIf(db is None, 'pyodbc is not installed')
    def test_add_scalar(self):
        expr = self.col_i + 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [6, 10, 18])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_sub_scalar(self):
        expr = self.col_i - 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 6, 14])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_mul_scalar(self):
        expr = self.col_i * 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [8, 16, 32])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_floordiv_scalar(self):
        expr = self.col_i // 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 4, 8])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_truediv_scalar(self):
        expr = self.col_i / 2
        result = eval(expr)
        self.assertEqual([float(x) for x in result], [2., 4., 8.])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_mod_scalar(self):
        expr = self.col_i % 3
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [1, 2, 1])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_neg_scalar(self):
        expr = -self.col_i
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [-4, -8, -16])

    ## compare

    @skipIf(db is None, 'pyodbc is not installed')
    def test_eq_scalar(self):
        expr = self.col_i == 8
        result = eval(expr)
        self.assertEqual(result.dshape.measure, bool_)
        self.assertEqual([bool(x) for x in result], [False, True, False])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_ne_scalar(self):
        expr = self.col_i != 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_lt_scalar(self):
        expr = self.col_i < 5
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, False])


    @skipIf(db is None, 'pyodbc is not installed')
    def test_le_scalar(self):
        expr = self.col_i <= 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, True, False])


    @skipIf(db is None, 'pyodbc is not installed')
    def test_gt_scalar(self):
        expr = self.col_i > 9
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, False, True])


    @skipIf(db is None, 'pyodbc is not installed')
    def test_ge_scalar(self):
        expr = self.col_i >= 8
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, True])

    ## logical

    @skipIf(db is None, 'pyodbc is not installed')
    def test_and(self):
        expr = blaze.logical_and(5 < self.col_i, self.col_i < 10)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, False])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_or(self):
        expr = blaze.logical_or(self.col_i < 5, self.col_i > 10)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_xor(self):
        expr = blaze.logical_xor(self.col_i < 9, self.col_i > 6)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [True, False, True])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_not(self):
        expr = blaze.logical_not(self.col_i < 5)
        result = eval(expr)
        self.assertEqual([bool(x) for x in result], [False, True, True])


class TestSQLUFuncExpressions(TestSQL):

    @skipIf(db is None, 'pyodbc is not installed')
    def test_select_expr(self):
        raise SkipTest("Correctly compose queries with aggregations")

        expr = ((ops.max(self.col_price) / ops.min(self.col_price)) *
                (self.col_i + 2) * 3.1 -
                ops.avg(self.col_i))
        result = eval(expr)

        np_result = ((np.max(self.np_price) / np.min(self.np_price)) *
                     (self.np_i + 2) * 3.1 -
                     np.average(self.np_i) / np.max(self.np_price))

        self.assertEqual([float(x) for x in result],
                         [float(x) for x in np_result])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_select_where(self):
        expr = ops.index(self.col_i + 2 * self.col_price,
                         blaze.logical_and(self.col_price > 5, self.col_price < 7))
        result = eval(expr)

        np_result = (self.np_i + 2 * self.np_price)[
            np.logical_and(self.np_price > 5, self.np_price < 7)]

        self.assertEqual([float(x) for x in result],
                         [float(x) for x in np_result])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_select_where2(self):
        expr = ops.index(self.col_i + 2 * self.col_price,
                         blaze.logical_or(
                             blaze.logical_and(self.col_price > 5,
                                               self.col_price < 7),
                             self.col_i > 6))
        result = eval(expr)

        np_result = (self.np_i + 2 * self.np_price)[
            np.logical_or(
                np.logical_and(self.np_price > 5,
                               self.np_price < 7),
                self.np_i > 6)]

        self.assertEqual([float(x) for x in result],
                         [float(x) for x in np_result])



class TestSQLDataTypes(TestSQL):

    @skipIf(db is None, 'pyodbc is not installed')
    def test_int(self):
        expr = self.col_i // 2
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [2, 4, 8])


class TestSQLColumns(TestSQL):

    @skipIf(db is None, 'pyodbc is not installed')
    def test_query(self):
        expr = add(self.col_i, self.col_i)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [8, 16, 32])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_query_scalar(self):
        expr = add(self.col_i, 2)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [6, 10, 18])

    @skipIf(db is None, 'pyodbc is not installed')
    def test_query_where(self):
        expr = ops.index(self.col_i + self.col_i, self.col_i > 5)
        result = eval(expr)
        self.assertEqual([int(x) for x in result], [16, 32])


class TestSQLTable(TestSQL):

    #@skipIf(db is None, 'pyodbc is not installed')
    @skip("there's an inconsistency between the table and column datashapes")
    def test_query_where(self):
        expr = ops.index(self.table, self.col_i > 5)
        result = eval(expr)
        row1, row2 = result
        self.assertEqual((int(row1[0]), str(row1[1]), float(row1[2])),
                         (8, "world", 4.2))
        self.assertEqual((int(row2[0]), str(row2[1]), float(row2[2])),
                         (16, "!", 8.4))

    @skipIf(db is None, 'pyodbc is not installed')
    def test_index_table(self):
        expr = self.table[:, 'i']
        self.assertEqual([int(i) for i in expr], [4, 8, 16])

    #@skipIf(db is None, 'pyodbc is not installed')
    @skip("there's an inconsistency between the table and column datashapes")
    def test_index_sql_result_table(self):
        expr = ops.index(self.table, self.col_i > 5)
        result = eval(expr)
        i_col = result[:, 'i']
        self.assertEqual([int(i_col[0]), int(i_col[1])], [8, 16])


class TestSQLStr(TestSQL):

    @skipIf(db is None, 'pyodbc is not installed')
    def test_str(self):
        repr(self.table)


if __name__ == '__main__':
    # TestSQLTable('test_query_where').debug()
    # TestSQLUFuncExpressions('test_select_where').debug()
    # TestSQLTable('test_index_sql_result_table').debug()
    # TestSQLStr('test_str').debug()
    unittest.main()
