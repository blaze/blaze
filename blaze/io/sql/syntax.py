# -*- coding: utf-8 -*-

"""
SQL syntax building.
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

Table   = namedtuple('Table',   ['tablename'])
Column  = namedtuple('Column',  ['table', 'colname'])
Select  = namedtuple('Select',  ['expr', 'from_expr', 'where', 'order'])
From    = namedtuple('From',    ['exprs'])
Where   = namedtuple('Where',   ['expr'])
OrderBy = namedtuple('OrderBy', ['expr'])
GroupBy = namedtuple('GroupBy', ['expr'])

class Expr(object):
    def __init__(self, *args):
        self.args = args

def reorder_select(query):
    """
    Reorder SQL query to prepare for codegen.
    """


def emit(q):
    """Emit SQL query from query object"""
    if isinstance(q, Table):
        return q.tablename
    if isinstance(q, Column):
        return "%s.%s" % (emit(q.table), emit(q.colname))
    elif isinstance(q, Select):
        return "SELECT %s %s %s %s" % (emit(q.expr),
                                       emit(q.from_expr),
                                       emit(q.where),
                                       emit(q.order))
    elif isinstance(q, From):
        return "FROM %s" % ", ".join(emit(expr) for expr in q.exprs)
    elif isinstance(q, Where):
        return "WHERE %s" % (emit(q.expr),)
    elif isinstance(q, OrderBy):
        return "ORDER BY %s" % (emit(q.expr),)
    elif isinstance(q, GroupBy):
        return "GROUP BY %s" % (emit(q.expr),)
    elif isinstance(q, Expr):
        return "(%s)" % " ".join(emit(arg) for arg in q.args)
    elif q is None:
        return ""
    else:
        return str(q)


if __name__ == '__main__':
    table = Table('Table')
    col1 = Column(table, 'attr1')
    col2 = Column(table, 'attr2')
    expr = Expr(Expr(col1, '+', col1), '-', col2)
    query = Select(expr, From(table), Where(Expr(col1, '=', col2)), None)
    print(emit(query))