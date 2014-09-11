from __future__ import absolute_import, division, print_function

import sys
from operator import itemgetter
import operator
from toolz import compose, identity
from collections import Iterator

from blaze.expr.table import *
from blaze.expr.table import count as Count
from . import core, python
from .python import (compute, rrowfunc, rowfunc, RowWise, pair_assemble,
                     reduce_by_funcs, binops, like_regex_predicate)
from ..compatibility import builtins, unicode
from ..expr import table
from ..dispatch import dispatch
from ..data.utils import listpack

from .core import compute, compute_one

from toolz.curried import get

__all__ = ['RDD', 'pyspark', 'SparkContext']

try:
    from pyspark import SparkContext
    import pyspark
    from pyspark.rdd import RDD
except ImportError:
    #Create a dummy RDD for py 2.6
    class Dummy(object):
        sum = max = min = count = distinct = mean = variance = stdev = None
    SparkContext = Dummy
    pyspark = Dummy()
    pyspark.rdd = Dummy()
    RDD = Dummy

# PySpark adds a SIGCHLD signal handler, but that breaks other packages, so we
# remove it
# See https://issues.apache.org/jira/browse/SPARK-1394
try:
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
except:
    pass


@dispatch(RowWise, RDD)
def compute_one(t, rdd, **kwargs):
    func = rowfunc(t)
    return rdd.map(func)


@dispatch(Selection, RDD)
def compute_one(t, rdd, **kwargs):
    predicate = rrowfunc(t.predicate, t.child)
    return rdd.filter(predicate)


rdd_reductions = {
        table.sum: RDD.sum,
        table.min: RDD.min,
        table.max: RDD.max,
        table.count: RDD.count,
        table.mean: RDD.mean,
        table.var: RDD.variance,
        table.std: RDD.stdev,
        table.nunique: compose(RDD.count, RDD.distinct)}


@dispatch(tuple(rdd_reductions), RDD)
def compute_one(t, rdd, **kwargs):
    return rdd_reductions[type(t)](rdd)


def istruthy(x):
    return not not x


@dispatch(table.any, RDD)
def compute_one(t, rdd, **kwargs):
    return istruthy(rdd.filter(identity).take(1))


@dispatch(table.all, RDD)
def compute_one(t, rdd, **kwargs):
    return not rdd.filter(lambda x: not x).take(1)


@dispatch(Head, RDD)
def compute_one(t, rdd, **kwargs):
    return rdd.take(t.n)


@dispatch(Sort, RDD)
def compute_one(t, rdd, **kwargs):
    if isinstance(t.key, (str, unicode, tuple, list)):
        key = rowfunc(t.child[t.key])
    else:
        key = rrowfunc(t.key, t.child)
    return (rdd.keyBy(key)
                .sortByKey(ascending=t.ascending)
                .map(lambda x: x[1]))


@dispatch(Distinct, RDD)
def compute_one(t, rdd, **kwargs):
    return rdd.distinct()


@dispatch(Join, RDD, RDD)
def compute_one(t, lhs, rhs, **kwargs):
    on_left = rowfunc(t.lhs[t.on_left])
    on_right = rowfunc(t.rhs[t.on_right])

    lhs = lhs.keyBy(on_left)
    rhs = rhs.keyBy(on_right)


    if t.how == 'inner':
        rdd = lhs.join(rhs)
    elif t.how == 'left':
        rdd = lhs.leftOuterJoin(rhs)
    elif t.how == 'right':
        rdd = lhs.rightOuterJoin(rhs)
    elif t.how == 'outer':
        # https://issues.apache.org/jira/browse/SPARK-546
        raise NotImplementedError("Spark does not yet support full outer join")

    assemble = pair_assemble(t)

    return rdd.map(lambda x: assemble(x[1]))


python_reductions = {
              table.sum: builtins.sum,
              table.count: builtins.len,
              table.max: builtins.max,
              table.min: builtins.min,
              table.any: builtins.any,
              table.all: builtins.all,
              table.mean: python._mean,
              table.var: python._var,
              table.std: python._std,
              table.nunique: lambda x: len(set(x))}

@dispatch(By, RDD)
def compute_one(t, rdd, **kwargs):
    if ((isinstance(t.apply, Reduction) and type(t.apply) in binops) or
        (isinstance(t.apply, Summary) and builtins.all(type(val) in binops
                                                for val in t.apply.values))):
        grouper, binop, combiner, initial = reduce_by_funcs(t)

        if t.grouper.iscolumn:
            keyfunc = lambda x: (x,)
        else:
            keyfunc = identity
        if isinstance(t.apply, Reduction):
            valfunc = lambda x: (x,)
        else:
            valfunc = identity
        unpack = lambda kv: keyfunc(kv[0]) + valfunc(kv[1])


        create = lambda v: binop(initial, v)

        return (rdd.keyBy(grouper)
                   .combineByKey(create, binop, combiner)
                   .map(unpack))


        d = reduceby(grouper, binop, seq, initial)
    else:
        raise NotImplementedError("By only implemented for common reductions."
                                  "\nGot %s" % type(t.apply))


@dispatch((Label, ReLabel), RDD)
def compute_one(t, rdd, **kwargs):
    return rdd


@dispatch(Summary, RDD)
def compute_one(t, rdd, **kwargs):
    rdd = rdd.cache()
    return tuple(compute(value, {t.child: rdd}) for value in t.values)


@dispatch(Like, RDD)
def compute_one(t, rdd, **kwargs):
    predicate = like_regex_predicate(t)
    return rdd.filter(predicate)


@dispatch(RDD, RDD)
def into(a, b):
    return b


@dispatch(SparkContext, (list, tuple, Iterator))
def into(sc, seq):
    return sc.parallelize(seq)


@dispatch(RDD, (list, tuple))
def into(rdd, seq):
    return into(rdd.context, seq)


@dispatch(list, RDD)
def into(seq, rdd):
    return rdd.collect()


@dispatch(Union, RDD, tuple)
def compute_one(t, _, children):
    return reduce(RDD.union, children)
