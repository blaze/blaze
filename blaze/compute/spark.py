from __future__ import absolute_import, division, print_function

from multipledispatch import dispatch
import sys
from operator import itemgetter

from blaze.expr.table import *
from blaze.expr.table import count as Count
from . import core, python
from .python import compute
from ..compatibility import builtins
from ..expr import table

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


def func_from_columnwise(t):
    columns = [t.parent.columns.index(arg.columns[0]) for arg in t.arguments]
    _func = eval(core.columnwise_funcstr(t))

    getter = itemgetter(*columns)
    def func(x):
        fields = getter(x)
        return func(*fields)

    return func


@dispatch(ColumnWise, pyspark.rdd.RDD)
def compute(t, rdd):
    if not all(isinstance(arg, Column) for arg in t.arguments):
        raise NotImplementedError("Expected arguments to be columns")
    parents = [arg.parent for arg in t.arguments]
    if not len(set(parents)) == 1:
        raise NotImplementedError("Columnwise defined only for single parent")

    rdd = compute(parents[0], rdd)

    func = func_from_columnwise(t)
    return rdd.map(func)


@dispatch(Selection, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.parent, rdd)
    print(core.columnwise_funcstr(t.predicate))
    _predicate = eval(core.columnwise_funcstr(t.predicate))
    predicate = lambda x: _predicate(*x)
    return rdd.filter(_predicate)


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


reductions = {table.sum: builtins.sum,
              table.count: builtins.len,
              table.max: builtins.max,
              table.min: builtins.min,
              table.any: builtins.any,
              table.all: builtins.all,
              table.mean: python._mean,
              table.var: python._var,
              table.std: python._std,
              table.nunique: lambda x: len(set(x))}


@dispatch(By, pyspark.rdd.RDD)
def compute(t, rdd):
    rdd = compute(t.parent, rdd)

    if not isinstance(t.grouper, Projection) and t.grouper.parent == t:
        raise NotImplementedError("By grouper must be projection of table")

    indices = [t.grouper.parent.columns.index(c) for c in t.grouper.columns]
    group_fn = itemgetter(*indices)

    try:
        reduction = reductions[type(t.apply)]
    except KeyError:
        raise NotImplementedError("By only implemented for common reductions."
                                  "\nGot %s" % type(t.apply))
    if isinstance(t.apply.parent, ColumnWise):
        pre_reduction = func_from_columnwise(t.apply.parent)
    elif isinstance(t.apply.parent, Column):
        pre_reduction = itemgetter(t.apply.parent.parent.columns.index(
                                        t.apply.parent.columns[0]))
    else:
        raise NotImplementedError("By only implemented for reductions of"
                                  " Columns or ColumnWises.\n"
                                  "Got: %s" % t.apply.parent)

    def pre_reduction_func(x):
        return group_fn(x), pre_reduction(x)

    return (rdd.map(pre_reduction_func)
               .groupByKey()
               .map(lambda x: (x[0], reduction(x[1]))))


@dispatch((Label, ReLabel), pyspark.rdd.RDD)
def compute(t, rdd):
    return compute(t.parent, rdd)
