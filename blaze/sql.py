from .compute.sql import *
from .compute.sql import select
from .data.sql import *
from .expr.table import Join, Expr, TableExpr
from .expr.scalar.core import Scalar

import sqlalchemy

@dispatch(Expr, SQL)
def compute_one(t, ddesc):
    return compute_one(t, ddesc.table)


@dispatch(Expr, sql.sql.ClauseElement, dict)
def post_compute(expr, query, d):
    try:
        engines = set([dd.engine for dd in d.values()])
    except:
        return query
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
