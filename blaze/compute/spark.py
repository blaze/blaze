from __future__ import absolute_import, division, print_function


from blaze.expr.table import *
from multipledispatch import dispatch
import pyspark
import itertools


@dispatch(Projection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.table, rdd)
    cols = [t.table.schema[0].names.index(col) for col in t.columns]
    return rdd.map(lambda x: [x[c] for c in cols])


@dispatch(Column, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.table, rdd)
    col_idx = t.table.schema[0].names.index(t.columns[0])
    return rdd.map(lambda x: x[col_idx])


@dispatch(TableSymbol, pyspark.rdd.RDD)
def compute(t, s):
    return s


@dispatch(Selection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.table, rdd)
    col_idx = t.table.schema[0].names.index(t.columns[0])
    return rdd.filter(lambda x: t.predicate.op(x[col_idx], t.predicate.rhs))


@dispatch(Join, pyspark.rdd.RDD, pyspark.rdd.RDD)
def compute(t, lhs, rhs):
    lhs = compute(t.lhs, lhs)
    rhs = compute(t.rhs, rhs)

    col_idx_lhs = t.lhs.schema[0].names.index(t.on_left)
    col_idx_rhs = t.rhs.schema[0].names.index(t.on_right)

    lhs = lhs.keyBy(lambda x: x[col_idx_lhs])
    rhs = rhs.keyBy(lambda x: x[col_idx_rhs])

    # Calculate the indices we want in the joined table
    lhs_indices = [1]*len(t.lhs.columns)
    rhs_indices = [1 if i != col_idx_rhs else 0 for i in range(0,
                   len(t.rhs.columns))]
    indices = lhs_indices + rhs_indices
    # Perform the spark join, then reassemple the table
    joined_rdd = lhs.join(rhs)
    out_rdd = joined_rdd.map(lambda x: [i for i in itertools.compress(
        itertools.chain.from_iterable(x[1]), indices)])
    return out_rdd
