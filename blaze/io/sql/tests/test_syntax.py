# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import unittest
from blaze.io.sql.syntax import (Table, Column, Select, Expr, Call, From, Where,
                                 GroupBy, OrderBy, emit)


def assert_query(result, query):
    assert " ".join(result.split()) == " ".join(query.split()), (result, query)

class TestSyntax(unittest.TestCase):

    def test_syntax_where(self):
        table = Table('Table')
        col1 = Column(table, 'attr1')
        col2 = Column(table, 'attr2')
        expr = Expr(Expr(col1, '+', col1), '-', col2)
        query = Select([expr],
                       From(table),
                       Where(Expr(col1, '=', col2)),
                       None, None)
        result = emit(query)
        assert_query(result, "SELECT ((Table.attr1 + Table.attr1) - Table.attr2) "
                             "FROM Table WHERE (Table.attr1 = Table.attr2)")

    def test_syntax_order(self):
        table = Table('Table')
        col1 = Column(table, 'attr1')
        col2 = Column(table, 'attr2')
        expr = Expr(col1, '+', col1)
        query = Select([expr],
                       From(table),
                       Where(Expr(col1, '=', col2)),
                       None,
                       OrderBy(col1))
        result = emit(query)
        assert_query(result, "SELECT (Table.attr1 + Table.attr1) "
                             "FROM Table "
                             "WHERE (Table.attr1 = Table.attr2) "
                             "ORDER BY Table.attr1 ASC")

    def test_syntax_groupby(self):
        table = Table('Table')
        col1 = Column(table, 'attr1')
        col2 = Column(table, 'attr2')
        query = Select([col1, Call('SUM', col2)],
                       From(table),
                       Where(Expr(col1, '=', col2)),
                       GroupBy([col1]),
                       None)
        result = emit(query)
        assert_query(result, "SELECT Table.attr1, SUM(Table.attr2) "
                             "FROM Table "
                             "WHERE (Table.attr1 = Table.attr2) "
                             "GROUP BY Table.attr1")


if __name__ == '__main__':
    unittest.main()