from __future__ import absolute_import, division, print_function

import datashape
from datashape import (dshape, DataShape, Record, isdimension, Option,
        discover, Tuple)

from .dispatch import dispatch

__all__ = []

try:
    import pyspark
    from pyspark import sql, RDD
    from pyspark.sql import (IntegerType, FloatType, StringType, TimestampType,
            StructType, StructField, ArrayType, SchemaRDD, SQLContext,
            ShortType, DoubleType, BooleanType, LongType)
    from pyspark import SparkContext
except ImportError:
    pyspark = None

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


if pyspark:

    types = {datashape.int16: ShortType(),
             datashape.int32: IntegerType(),
             datashape.int64: IntegerType(),
             datashape.float32: FloatType(),
             datashape.float64: DoubleType(),
             datashape.real: DoubleType(),
             datashape.time_: TimestampType(),
             datashape.date_: TimestampType(),
             datashape.datetime_: TimestampType(),
             datashape.bool_: BooleanType(),
             datashape.string: StringType()}

    rev_types = {IntegerType(): datashape.int64,
                 ShortType(): datashape.int32,
                 LongType(): datashape.int64,
                 FloatType(): datashape.float32,
                 DoubleType(): datashape.float64,
                 StringType(): datashape.string,
                 TimestampType(): datashape.datetime_,
                 BooleanType(): datashape.bool_}

    def sparksql_to_ds(ss):
        """ Convert datashape to SparkSQL type system

        >>> sparksql_to_ds(IntegerType())  # doctest: +SKIP
        ctype("int64")

        >>> sparksql_to_ds(ArrayType(IntegerType(), False))  # doctest: +SKIP
        dshape("var * int64")

        >>> sparksql_to_ds(ArrayType(IntegerType(), True))  # doctest: +SKIP
        dshape("var * ?int64")

        >>> sparksql_to_ds(StructType([  # doctest: +SKIP
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
            return dshape(Record([[field.name, Option(sparksql_to_ds(field.dataType))
                                        if field.nullable
                                        else sparksql_to_ds(field.dataType)]
                            for field in ss.fields]))
        raise NotImplementedError("SparkSQL type not known %s" % ss)


    def ds_to_sparksql(ds):
        """ Convert datashape to SparkSQL type system

        >>> print(ds_to_sparksql('int32')) # doctest: +SKIP
        IntegerType

        >>> print(ds_to_sparksql('5 * int32')) # doctest: +SKIP
        ArrayType(IntegerType,false)

        >>> print(ds_to_sparksql('5 * ?int32'))  # doctest: +SKIP
        ArrayType(IntegerType,true)

        >>> print(ds_to_sparksql('{name: string, amount: int32}'))  # doctest: +SKIP
        StructType(List(StructField(name,StringType,false),StructField(amount,IntegerType,false)))

        >>> print(ds_to_sparksql('10 * {name: string, amount: ?int32}'))  # doctest: +SKIP
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


    @dispatch(SQLContext, RDD)
    def into(sqlContext, rdd, schema=None, columns=None, **kwargs):
        """ Convert a normal PySpark RDD to a SparkSQL RDD

        Schema inferred by ds_to_sparksql.  Can also specify it explicitly with
        schema keyword argument.
        """
        schema = schema or discover(rdd).subshape[0]
        if isinstance(schema[0], Tuple):
            columns = columns or list(range(len(schema[0].dshapes)))
            types = schema[0].dshapes
            schema = dshape(Record(list(zip(columns, types))))
        sql_schema = ds_to_sparksql(schema)
        return sqlContext.applySchema(rdd, sql_schema)


    from blaze.expr import Expr, TableExpr
    @dispatch(SQLContext, (TableExpr, Expr, object))
    def into(sqlContext, o, **kwargs):
        schema = kwargs.pop('schema', None) or discover(o).subshape[0]
        return into(sqlContext, into(sqlContext._sc, o), schema=schema, **kwargs)


    @dispatch((tuple, list, set), SchemaRDD)
    def into(a, b, **kwargs):
        if not isinstance(a, type):
            a = type(a)
        return a(map(tuple, b.collect()))


    @dispatch(SchemaRDD)
    def discover(srdd):
        return datashape.var * sparksql_to_ds(srdd.schema())
