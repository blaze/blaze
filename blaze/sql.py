from __future__ import absolute_import, division, print_function

from .compute.sql import select
from .data.sql import SQL, dispatch, first
from .expr import Expr, TableExpr, Projection, Column, UnaryOp
from .expr.scalar.core import Scalar
from .compatibility import basestring
from .api.resource import resource


import sqlalchemy as sa

__all__ = ['compute_up', 'SQL']


@dispatch((Column, Projection, Expr, UnaryOp), SQL)
def compute_up(t, ddesc, **kwargs):
    return compute_up(t, ddesc.table, **kwargs)


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


@dispatch(SQL, basestring)
def create_index(s, column, name=None, unique=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    sa.Index(name, getattr(s.table.c, column), unique=unique).create(s.engine)


@dispatch(SQL, list)
def create_index(s, columns, name=None, unique=False):
    if name is None:
        raise ValueError('SQL indexes must have a name')
    args = name,
    args += tuple(getattr(s.table.c, column) for column in columns)
    sa.Index(*args, unique=unique).create(s.engine)

@resource.register('(sqlite|postgresql|mysql)://.*::\w*', priority=11)
def resource_sql_single_uri(uri, *args, **kwargs):
    uri, table_name = uri.rsplit('::', 1)
    return SQL(uri, table_name, *args, **kwargs)

@resource.register('(sqlite|postgresql|mysql)://.*')
def resource_sql(uri, table_name, *args, **kwargs):
    return SQL(uri, table_name, *args, **kwargs)
