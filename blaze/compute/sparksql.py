from __future__ import absolute_import, division, print_function

import datashape
from datashape import discover
from datashape.predicates import isrecord
import sqlalchemy as sa
import pandas as pd
from toolz import pipe, curry, valmap
from toolz.curried import filter, map
from into import convert
from into.backends.sql import dshape_to_alchemy

from ..dispatch import dispatch
from ..expr import Expr, Field, symbol
from .core import compute
from .utils import literalquery
from .spark import Dummy


__all__ = []


try:
    from pyspark.sql import SQLContext, DataFrame as SparkDataFrame
    from pyspark.sql import (ByteType, ShortType, IntegerType, LongType,
                             FloatType, DoubleType, StringType, BinaryType,
                             BooleanType, TimestampType, DateType)
    from pyhive.sqlalchemy_hive import HiveDialect
except ImportError:
    SparkDataFrame = SQLContext = Dummy
    ByteType = ShortType = IntegerType = LongType = FloatType = Dummy()
    DoubleType = StringType = BinaryType = BooleanType = Dummy()
    TimestampType = DateType = Dummy()


@convert.register(list, SparkDataFrame)
def sparksql_dataframe_to_list(df, **kwargs):
    return list(map(tuple, df.collect()))


@convert.register(pd.DataFrame, SparkDataFrame)
def sparksql_dataframe_to_pandas_dataframe(df, **kwargs):
    return pd.DataFrame(convert(list, df, **kwargs), columns=df.columns)


def make_sqlalchemy_table(expr):
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

    # get sqlalchemy tables
    sa_tables = map(make_sqlalchemy_table, tables)

    # compute using sqlalchemy
    scope = dict(zip(new_leaves, sa_tables))
    query = compute(expr, scope)

    # interpolate params
    compiled = literalquery(query, dialect=HiveDialect())
    return data.sql(str(compiled))


# see http://spark.apache.org/docs/latest/sql-programming-guide.html#spark-sql-datatype-reference
sparksql_to_dshape = {
    ByteType: datashape.int8,
    ShortType: datashape.int16,
    IntegerType: datashape.int32,
    LongType: datashape.int64,
    FloatType: datashape.float32,
    DoubleType: datashape.float64,
    StringType: datashape.string,
    BinaryType: datashape.bytes_,
    BooleanType: datashape.bool_,
    TimestampType: datashape.datetime_,
    DateType: datashape.date_,
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
    dshapes = valmap(discover, get_catalog(ctx))
    return datashape.DataShape(datashape.Record(dshapes.items()))


@dispatch(SparkDataFrame)
def discover(df):
    return datashape.var * schema_to_dshape(df.schema())
