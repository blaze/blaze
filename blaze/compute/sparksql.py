from __future__ import absolute_import, division, print_function

import toolz
from toolz import pipe
import itertools
from datashape import discover, Unit, Tuple, Record, iscollection, isscalar
import sqlalchemy as sa
from into.backends.sql import dshape_to_alchemy

from ..dispatch import dispatch
from ..expr import *
from .utils import literalquery

__all__ = []

try:
    import pyspark
    from pyspark.sql import SchemaRDD
except ImportError:
    SchemaRDD = type(None)

names = ('_table_%d' % i for i in itertools.count(1))

__all__ = []

class SparkSQLQuery(object):
    """ Pair of PySpark SQLContext and SQLAlchemy Table

    Python's SparkSQL interface only accepts strings.  We use SQLAlchemy to
    generate these strings.  To do this we'll have to pass around pairs of
    (SQLContext, sqlalchemy.Selectable).  Additionally we track a mapping of
    {schemardd: sqlalchemy.Table}

    Parameters
    ----------

    context: pyspark.sql.SQLContext

    query: sqlalchemy.Selectable

    mapping: dict :: {pyspark.sql.SchemaRDD: sqlalchemy.Table}
    """
    __slots__ = 'context', 'query', 'mapping'

    def __init__(self, context, query, mapping):
        self.context = context
        self.query = query
        self.mapping = mapping


def make_query(rdd, primary_key='', name=None):
    # SparkSQL
    name = name or next(names)
    context = rdd.sql_ctx
    context.registerRDDAsTable(rdd, name)

    # SQLAlchemy
    schema = discover(rdd).subshape[0]
    columns = dshape_to_alchemy(schema)
    for column in columns:
        if column.name == primary_key:
            column.primary_key = True

    metadata = sa.MetaData()  # TODO: sync this between many tables

    query = sa.Table(name, metadata, *columns)

    mapping = {rdd: query}

    return SparkSQLQuery(context, query, mapping)


@dispatch(Symbol, SchemaRDD)
def compute_up(ts, rdd, **kwargs):
    return make_query(rdd)


@dispatch((var, Label, std, Sort, count, nunique, Selection, mean,
           Head, ReLabel, Distinct, ElemWise, By, any, all, sum, max,
           min, Reduction, Projection, Field), SchemaRDD)
def compute_up(e, rdd, **kwargs):
    return compute_up(e, make_query(rdd), **kwargs)


@dispatch((BinOp, Join),
          (SparkSQLQuery, SchemaRDD),
          (SparkSQLQuery, SchemaRDD))
def compute_up(e, a, b, **kwargs):
    if not isinstance(a, SparkSQLQuery):
        a = make_query(a)
    if not isinstance(b, SparkSQLQuery):
        b = make_query(b)
    return compute_up(e, a, b, **kwargs)


@dispatch((UnaryOp, Expr), SparkSQLQuery)
def compute_up(expr, q, **kwargs):
    scope = kwargs.pop('scope', dict())
    scope = dict((t, q.mapping.get(data, data)) for t, data in scope.items())

    q2 = compute_up(expr, q.query, scope=scope, **kwargs)
    return SparkSQLQuery(q.context, q2, q.mapping)


@dispatch((BinOp, Join, Expr), SparkSQLQuery, SparkSQLQuery)
def compute_up(expr, a, b, **kwargs):
    assert a.context == b.context

    mapping = toolz.merge(a.mapping, b.mapping)

    scope = kwargs.pop('scope', dict())
    scope = dict((t, mapping.get(data, data)) for t, data in scope.items())

    c = compute_up(expr, a.query, b.query, scope=scope, **kwargs)
    return SparkSQLQuery(a.context, c, mapping)


from .sql import select
def sql_string(query):
    return pipe(query, select, literalquery, str)

@dispatch(Expr, SparkSQLQuery)
def post_compute(expr, query, scope=None):
    result = query.context.sql(sql_string(query.query))
    if iscollection(expr.dshape) and isscalar(expr.dshape.measure):
        result = result.map(lambda x: x[0])
    return result


@dispatch(Head, SparkSQLQuery)
def post_compute(expr, query, scope=None):
    result = query.context.sql(sql_string(query.query))
    if iscollection(expr.dshape) and isscalar(expr.dshape.measure):
        result = result.map(lambda x: x[0])
    return result.collect()
