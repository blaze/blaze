from __future__ import absolute_import, division, print_function

from datashape.predicates import isscalar
from .compute.sql import select
from .data.sql import SQL, dispatch
from .expr import Expr, Projection, Field, UnaryOp, BinOp, Join
from .data.sql import SQL, dispatch
from .expr.scalar.core import Scalar
from .compatibility import basestring
from .api.resource import resource
from toolz import first


import sqlalchemy as sa

__all__ = ['compute_up', 'SQL']


@dispatch((Field, Projection, Expr, UnaryOp), SQL)
def compute_up(t, ddesc, **kwargs):
    return compute_up(t, ddesc.table, **kwargs)

@dispatch((BinOp, Join), SQL, sa.sql.Selectable)
def compute_up(t, lhs, rhs, **kwargs):
    return compute_up(t, lhs.table, rhs, **kwargs)

@dispatch((BinOp, Join), sa.sql.Selectable, SQL)
def compute_up(t, lhs, rhs, **kwargs):
    return compute_up(t, lhs, rhs.table, **kwargs)

@dispatch((BinOp, Join), SQL, SQL)
def compute_up(t, lhs, rhs, **kwargs):
    return compute_up(t, lhs.table, rhs.table, **kwargs)


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
    if isscalar(expr.dshape.measure):
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

@resource.register('(sqlite|postgresql|mysql|mysql\+pymysql)://.*')
def resource_sql(uri, table_name, *args, **kwargs):
    return SQL(uri, table_name, *args, **kwargs)


@resource.register('impala://.*')
def resource_sql(uri, table_name, *args, **kwargs):
    try:
        import impala.sqlalchemy
    except ImportError:
        raise ImportError("Please install or update `impyla` library")
    return SQL(uri, table_name, *args, **kwargs)
