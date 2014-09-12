
import pytest
pyspark = pytest.importorskip('pyspark')
from pyspark.sql import *
from blaze.sparksql import sparksql_to_ds, ds_to_sparksql
from datashape import *

def test_sparksql_to_ds():
    assert sparksql_to_ds(IntegerType()) == int64

    assert sparksql_to_ds(ArrayType(IntegerType(), False)) == dshape("var * int64")

    assert sparksql_to_ds(ArrayType(IntegerType(), True)) == dshape("var * ?int64")

    assert sparksql_to_ds(StructType([
                             StructField('name', StringType(), False),
                             StructField('amount', IntegerType(), True)])) \
                == dshape("{ name : string, amount : ?int64 }")


def test_ds_to_sparksql():
    assert ds_to_sparksql('int32') == IntegerType()

    assert ds_to_sparksql('5 * int32') == ArrayType(IntegerType(), False)

    assert ds_to_sparksql('5 * ?int32') == ArrayType(IntegerType(), True)

    assert ds_to_sparksql('{name: string, amount: int32}') == \
        StructType([StructField('name', StringType(), False),
                    StructField('amount', IntegerType(), False)])

    assert ds_to_sparksql('10 * {name: string, amount: ?int32}') == \
            ArrayType(StructType(
                        [StructField('name', StringType(), False),
                        StructField('amount', IntegerType(), True)]),
                        False)
