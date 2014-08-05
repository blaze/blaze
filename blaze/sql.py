from __future__ import absolute_import, division, print_function

from .compute.sql import *
from .compute.sql import select
from .data.sql import *
from .expr.table import Join, Expr, TableExpr, Projection, Column
from .expr.scalar.core import Scalar

import sqlalchemy

@dispatch((Column, Projection, Expr), SQL)
def compute_one(t, ddesc, **kwargs):
    return compute_one(t, ddesc.table, **kwargs)


@dispatch(Expr, sql.sql.ClauseElement, dict)
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
