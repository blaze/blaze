import pyspark
from pyspark import sql

import datashape
from datashape import dshape, DataShape, Record, isdimension, Option

types = {datashape.int32: sql.IntegerType(),
         datashape.int64: sql.IntegerType(),
         datashape.float32: sql.FloatType(),
         datashape.float64: sql.FloatType(),
         datashape.real: sql.FloatType(),
         datashape.time_: sql.TimestampType(),
         datashape.string: sql.StringType()}

def deoption(ds):
    if isinstance(ds, Option):
        return ds.ty
    else:
        return ds

def ds_to_sparksql(ds):
    """ Convert datashape to SparkSQL type system

    >>> print ds_to_sparksql('int32')
    IntegerType

    >>> print ds_to_sparksql('5 * int32')
    ArrayType(IntegerType,false)

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
