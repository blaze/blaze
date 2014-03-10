from __future__ import print_function, division, absolute_import

import unittest

from blaze.io.sql.syntax import (Table, Column, Select, Expr, Call, From, Where,
                                 GroupBy, OrderBy, qmap, emit,
                                 reorder_select, QWhere, QGroupBy, QOrderBy)


def assert_query(result, query):
    assert " ".join(result.split()) == " ".join(query.split()), (result, query)

table = Table('Table')
col1 = Column(table, 'attr1')
col2 = Column(table, 'attr2')


class TestSyntax(unittest.TestCase):

    def test_syntax_where(self):
        expr = Expr([Expr([col1, '+', col1]), '-', col2])
        query = Select([expr],
                       From([table]),
                       Where(Expr([col1, '=', col2])),
                       None, None)
        result = emit(query)
        assert_query(result, "SELECT ((Table.attr1 + Table.attr1) - Table.attr2) "
                             "FROM Table WHERE (Table.attr1 = Table.attr2)")

    def test_syntax_order(self):
        expr = Expr([col1, '+', col1])
        query = Select([expr],
                       From([table]),
                       Where(Expr([col1, '=', col2])),
                       None,
                       OrderBy([col1], True))
        result = emit(query)
        assert_query(result, "SELECT (Table.attr1 + Table.attr1) "
                             "FROM Table "
                             "WHERE (Table.attr1 = Table.attr2) "
                             "ORDER BY Table.attr1 ASC")

    def test_syntax_groupby(self):
        query = Select([col1, Call('SUM', [col2])],
                       From([table]),
                       Where(Expr([col1, '=', col2])),
                       GroupBy([col1]),
                       None)
        result = emit(query)
        assert_query(result, "SELECT Table.attr1, SUM(Table.attr2) "
                             "FROM Table "
                             "WHERE (Table.attr1 = Table.attr2) "
                             "GROUP BY Table.attr1")

    def test_qmap(self):
        query = Select([col1, Call('SUM', col2)],
                       From([table]),
                       Where(Expr([col1, '=', col2])),
                       GroupBy([col1]),
                       None)

        terms = []

        def f(q):
            terms.append(q)
            return q

        qmap(f, query)


class TestReorder(unittest.TestCase):

    def test_reorder_where(self):
        expr = QWhere(col1, Expr([col1, '<', col2]))
        query = reorder_select(expr)
        assert_query(emit(query),
                     "SELECT Table.attr1 FROM Table WHERE "
                     "(Table.attr1 < Table.attr2)")

    def test_reorder_groupby(self):
        expr = QGroupBy(QWhere(col1, Expr([col1, '<', col2])), [col2])
        query = reorder_select(expr)
        assert_query(emit(query),
                     "SELECT Table.attr1 "
                     "FROM Table "
                     "WHERE (Table.attr1 < Table.attr2) "
                     "GROUP BY Table.attr2 ")

    def test_reorder_orderby(self):
        expr = QOrderBy(
                    QGroupBy(
                        QWhere(col1,
                               Expr([col1, '<', col2])),
                        [col2]),
                    [Call('SUM', [col1])],
                    True)
        query = reorder_select(expr)
        assert_query(emit(query),
                     "SELECT Table.attr1 "
                     "FROM Table "
                     "WHERE (Table.attr1 < Table.attr2) "
                     "GROUP BY Table.attr2 "
                     "ORDER BY SUM(Table.attr1) ASC")



if __name__ == '__main__':
    unittest.main()
