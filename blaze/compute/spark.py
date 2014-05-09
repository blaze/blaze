from __future__ import absolute_import, division, print_function

from blaze.expr.table import *
from multipledispatch import dispatch
import pyspark

@dispatch(Projection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.table, rdd)
    print("T SCHEMA IS " + str(t.table.schema))
    cols = [ t.table.schema[0].names.index(col) for col in t.columns]
    print("COLS: " + str(cols))
    return rdd.map(lambda x: [x[c] for c in cols])

@dispatch(Column, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.table, rdd)
    col_idx =  t.table.schema[0].names.index(t.columns[0]) 
    return rdd.map(lambda x: x[col_idx])

@dispatch(TableSymbol, pyspark.rdd.RDD)
def compute(t, s):
    return s
