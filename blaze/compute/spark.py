from __future__ import absolute_import, division, print_function


from blaze.expr.table import *
from blaze.expr.table import count as Count
from blaze.compute.python import *
from multipledispatch import dispatch
import sys
try:
    from itertools import compress, chain
    import pyspark
except ImportError:
    #Create a dummy pyspark.rdd.RDD for py 2.6
    class Dummy(object):
        pass
    pyspark = Dummy()
    pyspark.rdd = Dummy()
    pyspark.rdd.RDD = Dummy

# PySpark adds a SIGCHLD signal handler, but that breaks other packages, so we
# remove it
# See https://issues.apache.org/jira/browse/SPARK-1394
try:
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
except:
    pass


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


def _close(fn, ap):
    """ Build a closure around a compute fn and an apply

    PySpark serialization, accompanied with some weirdness with Pandas and
    NumExpr force a kludgy solution to avoid serialization issues.

    See https://issues.apache.org/jira/browse/SPARK-1394
    """
    def _(x):
        return x[0], fn(ap, list(x[1]))
    return _


@dispatch(By, pyspark.rdd.RDD)
def compute(t, rdd):
    parent = compute(t.parent, rdd)
    key_by_idx = t.parent.schema[0].names.index(t.grouper.columns[0])
    keyed_rdd = parent.keyBy(lambda x: x[key_by_idx])
    grouped = keyed_rdd.groupByKey()
    compute_fn = compute.resolve((type(t.apply), list))
    f = _close(compute_fn, t.apply)
    return grouped.map(f)
