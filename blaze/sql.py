from __future__ import absolute_import, division, print_function

from .compute.sql import select
from .data.sql import SQL, dispatch, first
from .expr.table import Expr, TableExpr, Projection, Column
from .expr.scalar.core import Scalar
from .compatibility import basestring


import sqlalchemy as sa

__all__ = ['compute_one', 'SQL']


@dispatch((Column, Projection, Expr), SQL)
def compute_one(t, ddesc, **kwargs):
    return compute_one(t, ddesc.table, **kwargs)


@dispatch(Expr, sa.sql.ClauseElement, dict)
def post_compute(expr, query, d):
    """ Execute SQLAlchemy query against SQLAlchemy engines

    If the result of compute is a SQLAlchemy query then it is likely that the
    data elements are themselves SQL objects which contain SQLAlchemy engines.
    We find these engines and, if they are all the same, run the query against
    these engines and return the result.
    """
    if not all(isinstance(val, SQL) for val in d.values()):
        return query

    engines = set([dd.engine for dd in d.values()])

    if len(set(map(str, engines))) != 1:
        raise NotImplementedError("Expected single SQLAlchemy engine")

    engine = first(engines)

    with engine.connect() as conn:  # Perform query
        result = conn.execute(select(query)).fetchall()

    if isinstance(expr, Scalar):
        return result[0][0]
    if isinstance(expr, TableExpr) and expr.iscolumn:
        return [x[0] for x in result]
    return result


@dispatch(SQL)
def drop(s):
    s.table.drop(s.engine)


@dispatch(SQL, basestring, basestring)
def create_index(s, index_name, column_name, unique=False):
    sa.Index(index_name, getattr(s.table.c, column_name),
             unique=unique).create(s.engine)


@dispatch(SQL, basestring, list)
def create_index(s, index_name, column_names, unique=False):
    args = index_name,
    args += tuple(getattr(s.table.c, column_name)
                  for column_name in column_names)
    sa.Index(*args, unique=unique).create(s.engine)


@dispatch(SQL, dict)
def create_index(s, index_pairs, unique=False):
    column_names = list(index_pairs.values())
    assert all(hasattr(s.table.c, col_name) for col_name in column_names), \
        ('table %s does not have all input columns %s' % (s.table,
                                                          column_names))
    for index_name, column_name in index_pairs.items():
        create_index(s, index_name, column_name)
