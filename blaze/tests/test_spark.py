from blaze.spark import coerce
import pytest

pyspark = pytest.importorskip('pyspark')
sc = pyspark.SparkContext("local", "Spark app")


def test_spark_coerce():
    rdd = sc.parallelize([('1', 'hello'), ('2', 'world')])
    assert (coerce('{x: int, y: string}', rdd).collect() ==
            [(1, 'hello'), (2, 'world')])
