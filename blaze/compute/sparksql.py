from __future__ import absolute_import, division, print_function

import datashape
from datashape import discover
from datashape.predicates import isrecord
import sqlalchemy as sa
from toolz import pipe, curry
from toolz.curried import filter
from into import convert
from into.backends.sql import dshape_to_alchemy

from ..dispatch import dispatch
from ..expr import Expr, Field, symbol
from .core import compute
from .utils import literalquery


__all__ = []

from pyspark.sql import SQLContext
import pyspark.sql as sql


@convert.register(list, sql.DataFrame)
def sparksql_dataframe_to_list(df, **kwargs):
    return list(map(tuple, df.collect()))


@dispatch(Field, SQLContext)
def compute_up(expr, data, **kwargs):
    name = expr._name
    columns = dshape_to_alchemy(expr.dshape)
    return sa.Table(name, sa.MetaData(), *columns)


@curry
def istable(db, t):
    return (isinstance(t, Field) and isrecord(t.dshape.measure) and
            t._child.isidentical(db))


@dispatch(Expr, SQLContext)
def compute_down(expr, data):
    """ Compile a blaze expression to a sparksql expression"""
    leaves = expr._leaves()

    # make sure we only have a single leaf node
    if len(leaves) != 1:
        raise ValueError('Must compile from exactly one root database')

    leaf, = leaves

    # field expressions on the database are Field instances with a record
    # measure whose immediate child is the database leaf
    tables = pipe(expr._subterms(), filter(istable(leaf)), list)

    # raise if we don't have tables in our database
    if not tables:
        raise ValueError('Expressions not referencing a table cannot be '
                         'compiled')

    # make new symbols for each table
    new_leaves = [symbol(t._name, t.dshape) for t in tables]

    # sub them in the expression
    expr = expr._subs(dict(zip(tables, new_leaves)))

    # get sqlalchemy tables, we can't go through compute_down here as that will
    # recurse back into this function
    sa_tables = [compute_up(t, data) for t in tables]

    # compute using sqlalchemy
    scope = dict(zip(new_leaves, sa_tables))
    query = compute(expr, scope)

    # interpolate params
    qs = str(literalquery(query))
    return data.sql(qs)


# see http://spark.apache.org/docs/latest/sql-programming-guide.html#spark-sql-datatype-reference
sparksql_to_dshape = {
    sql.ByteType: datashape.int8,
    sql.ShortType: datashape.int16,
    sql.IntegerType: datashape.int32,
    sql.LongType: datashape.int64,
    sql.FloatType: datashape.float32,
    sql.DoubleType: datashape.float64,
    sql.StringType: datashape.string,
    sql.BinaryType: datashape.bytes_,
    sql.BooleanType: datashape.bool_,
    sql.TimestampType: datashape.datetime_,
    sql.DateType: datashape.date_,
    # sql.ArrayType: ?,
    # sql.MapTYpe: ?,
    # sql.StructType: ?
}


def schema_to_dshape(schema):
    dshape = []
    for field in schema.fields:
        name = field.name
        value = sparksql_to_dshape.get(type(field.dataType))
        if value is None:
            raise ValueError('No known mapping for SparkSQL type %r to '
                             'datashape type' % str(field.dataType))
        dshape.append((name, value))
    return datashape.Record(dshape)


def scala_set_to_set(ctx, x):
    from py4j.java_gateway import java_import

    # import scala
    java_import(ctx._jvm, 'scala')

    # grab Scala's set converter and convert to a Python set
    return set(ctx._jvm.scala.collection.JavaConversions.setAsJavaSet(x))


def get_catalog(ctx):
    # the .catalog() method yields a SimpleCatalog instance. This class is
    # hidden from the Python side, but is present in the public Scala API
    java_names = ctx._ssql_ctx.catalog().tables().keySet()
    table_names = scala_set_to_set(ctx, java_names)
    tables = map(ctx.table, table_names)
    return dict(zip(table_names, tables))


@dispatch(SQLContext)
def discover(ctx):
    catalog = get_catalog(ctx)
    tableshapes = [(name, datashape.var * schema_to_dshape(t.schema()))
                   for name, t in catalog.items()]
    return datashape.DataShape(datashape.Record(tableshapes))
