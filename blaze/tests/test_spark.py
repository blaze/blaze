from blaze.spark import *
import nose

try:
    import pyspark
    sc = pyspark.SparkContext("local", "Spark app")
except ImportError:
    raise nose.SkipTest('pyspark not available')


def test_spark_coerce():
    rdd = sc.parallelize([('1', 'hello'), ('2', 'world')])
    assert coerce('{x: int, y: string}', rdd).collect() == \
            [(1, 'hello'), (2, 'world')]
