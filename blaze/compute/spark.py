from __future__ import absolute_import, division, print_function


from blaze.expr.table import *
from blaze.expr.table import count as Count
from blaze.compute.python import *
from multipledispatch import dispatch
import pyspark
from itertools import compress, chain


@dispatch(Projection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.parent, rdd)
    cols = [t.parent.schema[0].names.index(col) for col in t.columns]
    return rdd.map(lambda x: [x[c] for c in cols])


@dispatch(Column, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.parent, rdd)
    col_idx = t.parent.schema[0].names.index(t.columns[0])
    return rdd.map(lambda x: x[col_idx])


@dispatch(TableSymbol, pyspark.rdd.RDD)
def compute(t, s):
    return s


@dispatch(Selection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.parent, rdd)
    col_idx = t.parent.schema[0].names.index(t.columns[0])
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
    reassemble = lambda x: list(compress(chain.from_iterable(x[1]), indices))
    out_rdd = lhs.join(rhs).map(reassemble)
    return out_rdd

@dispatch(By, pyspark.rdd.RDD)
def compute(t, rdd):
    parent = compute(t.parent, rdd)
    key_by_idx = t.parent.schema[0].names.index(t.grouper.columns[0])
    keyed_rdd = parent.keyBy(lambda x: x[key_by_idx])
    grouped = keyed_rdd.groupByKey()
    compute_fn = compute.resolve((type(t.apply), list))
    return grouped.map(lambda x: (x[0], compute_fn(t.apply, x[1])))


@dispatch(TableExpr, pyspark.rdd.ResultIterable)
def compute(t, ri):
    return compute(t, list(ri))
