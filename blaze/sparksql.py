import pyspark
from pyspark import sql
from pyspark.sql import (IntegerType, FloatType, StringType, TimestampType,
        StructType, StructField, ArrayType)

import datashape
from datashape import (dshape, DataShape, Record, isdimension, Option,
        discover, Tuple)

from .dispatch import dispatch

types = {datashape.int32: sql.IntegerType(),
         datashape.int64: sql.IntegerType(),
         datashape.float32: sql.FloatType(),
         datashape.float64: sql.FloatType(),
         datashape.real: sql.FloatType(),
         datashape.time_: sql.TimestampType(),
         datashape.date_: sql.TimestampType(),
         datashape.datetime_: sql.TimestampType(),
         datashape.string: sql.StringType()}

rev_types = {sql.IntegerType(): datashape.int64,
             sql.FloatType(): datashape.float64,
             sql.StringType(): datashape.string,
             sql.TimestampType(): datashape.datetime_}

def deoption(ds):
    """

    >>> deoption('int32')
    ctype("int32")

    >>> deoption('?int32')
    ctype("int32")
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and not isdimension(ds[0]):
        return deoption(ds[0])
    if isinstance(ds, Option):
        return ds.ty
    else:
        return ds


def sparksql_to_ds(ss):
    """ Convert datashape to SparkSQL type system

    >>> sparksql_to_ds(IntegerType())
    ctype("int64")

    >>> sparksql_to_ds(ArrayType(IntegerType(), False))
    dshape("var * int64")

    >>> sparksql_to_ds(ArrayType(IntegerType(), True))
    dshape("var * ?int64")

    >>> sparksql_to_ds(StructType([
    ...                         StructField('name', StringType(), False),
    ...                         StructField('amount', IntegerType(), True)]))
    dshape("{ name : string, amount : ?int64 }")
    """
    if ss in rev_types:
        return rev_types[ss]
    if isinstance(ss, ArrayType):
        elem = sparksql_to_ds(ss.elementType)
        if ss.containsNull:
            return datashape.var * Option(elem)
        else:
            return datashape.var * elem
    if isinstance(ss, StructType):
        return Record([[field.name, Option(sparksql_to_ds(field.dataType))
                                    if field.nullable
                                    else sparksql_to_ds(field.dataType)]
                        for field in ss.fields])
    raise NotImplementedError("SparkSQL type not known %s" % ss)

def ds_to_sparksql(ds):
    """ Convert datashape to SparkSQL type system

    >>> print ds_to_sparksql('int32')
    IntegerType

    >>> print ds_to_sparksql('5 * int32')
    ArrayType(IntegerType,false)

    >>> print ds_to_sparksql('5 * ?int32')
    ArrayType(IntegerType,true)

    >>> print ds_to_sparksql('{name: string, amount: int32}')
    StructType(List(StructField(name,StringType,false),StructField(amount,IntegerType,false)))

    >>> print ds_to_sparksql('10 * {name: string, amount: ?int32}')
    ArrayType(StructType(List(StructField(name,StringType,false),StructField(amount,IntegerType,true))),false)
    """
    if isinstance(ds, str):
        return ds_to_sparksql(dshape(ds))
    if isinstance(ds, Record):
        return sql.StructType([
            sql.StructField(name,
                            ds_to_sparksql(deoption(typ)),
                            isinstance(typ, datashape.Option))
            for name, typ in ds.fields])
    if isinstance(ds, DataShape):
        if isdimension(ds[0]):
            elem = ds.subshape[0]
            if isinstance(elem, DataShape) and len(elem) == 1:
                elem = elem[0]
            return sql.ArrayType(ds_to_sparksql(deoption(elem)),
                                 isinstance(elem, Option))
        else:
            return ds_to_sparksql(ds[0])
    if ds in types:
        return types[ds]
    raise NotImplementedError()


@dispatch(pyspark.sql.SchemaRDD, pyspark.RDD)
def into(_, rdd, schema=None, **kwargs):
    """ Convert a normal PySpark RDD to a SparkSQL RDD

    Schema inferred by ds_to_sparksql.  Can also specify it explicitly with
    schema keyword argument.
    """
    schema = schema or discover(rdd)
    sql_schema = ds_to_sparksql(schema).subshape[0]
    return sqlContext.applySchema(rdd, sql_schema)
