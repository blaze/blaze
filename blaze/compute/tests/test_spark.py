from __future__ import absolute_import, division, print_function

from blaze.compute.spark import *
from blaze.expr.table import *

import os
import platform
import pyspark
from pyspark import SparkConf, SparkContext

t = TableSymbol('{name: string, amount: int, id: int}')

data = [['Alice', 100, 1],
        ['Bob', 200, 2],
        ['Alice', 50, 3]]

sc = SparkContext("local", "Simple App")

rdd = sc.parallelize(data)

def test_table():
    assert compute(t, rdd) == rdd


def test_projection():
    assert compute(t['name'], rdd).collect() == rdd.map(lambda x: x[0]).collect()

def test_multicols_projection():
    assert compute(t[['amount', 'name']], rdd).collect() == [[100, 'Alice'], [200, 'Bob'], [50, 'Alice']]
