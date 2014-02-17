# -*- coding: utf-8 -*-

"""
SQL syntax building.
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

#------------------------------------------------------------------------
# Syntax (declarative)
#------------------------------------------------------------------------

def qtuple(name, attrs):
    cls = namedtuple(name, attrs)
    cls.__str__ = lambda self: "Query(%s)" % (emit(self),)
    return cls

Table   = qtuple('Table',   ['tablename'])
Column  = qtuple('Column',  ['table', 'colname'])
Select  = qtuple('Select',  ['exprs', 'from_expr', 'where',
                                 'groupby', 'order'])
Where   = qtuple('Where',   ['expr'])
GroupBy = qtuple('GroupBy', ['cols'])
From    = qtuple('From',    ['exprs'])
OrderBy = qtuple('OrderBy', ['exprs', 'ascending'])
Call    = qtuple('Call',    ['name', 'args'])
Expr    = qtuple('Expr',    ['args'])
And     = lambda e1, e2: Expr([e1, 'AND', e2])
Or      = lambda e1, e2: Expr([e1, 'OR', e2])
Not     = lambda e1: Expr(['NOT', e1])


def qmap(f, q):
    """
    Apply `f` post-order to all sub-terms in query term `q`.
    """
    if hasattr(q, '_fields'):
        attrs = []
        for field in q._fields:
            attr = getattr(q, field)
            attrs.append(qmap(f, attr))

        cls = type(q)
        obj = cls(*attrs)
        return f(obj)

    elif isinstance(q, (list, tuple)):
        cls = type(q)
        return cls(qmap(f, x) for x in q)

    return f(q)

#------------------------------------------------------------------------
# Query expressions
#------------------------------------------------------------------------

# These may be nested in an expression-like fashion. These expressions may
# then be reordered to obtain declarative syntax above

QWhere      = namedtuple('QWhere', ['arr', 'expr'])
QGroupBy    = namedtuple('QGroupBy', ['arr', 'keys'])
QOrderBy    = namedtuple('QOrderBy', ['arr', 'exprs', 'ascending'])

def reorder_select(query):
    """
    Reorder SQL query to prepare for codegen.
    """
    ## Extract info ##
    selects = []
    wheres = []
    orders = []
    groupbys = []
    tables = set()

    def extract(expr):
        if isinstance(expr, QWhere):
            selects.append(expr.arr)
            wheres.append(expr.expr)
            return expr.arr

        elif isinstance(expr, QGroupBy):
            selects.append(expr.arr)
            groupbys.extend(expr.keys)
            return expr.arr

        elif isinstance(expr, QOrderBy):
            selects.append(expr.arr)
            orders.append(expr)
            return expr.arr

        elif isinstance(expr, Table):
            tables.add(expr)

        return expr

    expr = qmap(extract, query)

    ## Build SQL syntax ##
    if isinstance(expr, Table):
        expr = '*'

    if len(orders) > 1:
        raise ValueError("Only a single ordering may be specified")
    elif orders:
        [order] = orders

    return Select([expr],
                  From(list(tables)),
                  Where(reduce(And, wheres)) if wheres else None,
                  GroupBy(groupbys) if groupbys else None,
                  OrderBy(order.exprs, order.ascending) if orders else None,
                  )

#------------------------------------------------------------------------
# Query generation
#------------------------------------------------------------------------

def emit(q):
    """Emit SQL query from query object"""
    if isinstance(q, Table):
        return q.tablename
    if isinstance(q, Column):
        return "%s.%s" % (emit(q.table), emit(q.colname))
    elif isinstance(q, Select):
        return "SELECT %s %s %s %s %s" % (
                    ", ".join(emit(expr) for expr in q.exprs),
                    emit(q.from_expr),
                    emit(q.where),
                    emit(q.groupby),
                    emit(q.order))
    elif isinstance(q, From):
        return "FROM %s" % ", ".join(emit(expr) for expr in q.exprs)
    elif isinstance(q, Where):
        return "WHERE %s" % (emit(q.expr),)
    elif isinstance(q, OrderBy):
        order_clause = "ORDER BY %s" % " ".join(emit(expr) for expr in q.exprs)
        return "%s %s" % (order_clause, "ASC" if q.ascending else "DESC")
    elif isinstance(q, GroupBy):
        return "GROUP BY %s" % ", ".join(emit(col) for col in q.cols)
    elif isinstance(q, Expr):
        return "(%s)" % " ".join(emit(arg) for arg in q.args)
    elif isinstance(q, Call):
        return "%s(%s)" % (q.name, ", ".join(emit(arg) for arg in q.args))
    elif q is None:
        return ""
    else:
        return str(q)


if __name__ == '__main__':
    table = Table('Table')
    col1 = Column(table, 'attr1')
    col2 = Column(table, 'attr2')
    expr = Expr(Expr(col1, '+', col1), '-', col2)
    query = Select([expr], From(table), Where(Expr(col1, '=', col2)),
                   None, None)
    print(emit(query))
