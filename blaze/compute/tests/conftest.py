import pytest


@pytest.fixture(scope='session')
def sc():
    pyspark = pytest.importorskip('pyspark')
    return pyspark.SparkContext('local', 'blaze')
