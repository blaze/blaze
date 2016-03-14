from __future__ import absolute_import, division, print_function

import fnmatch

from toolz import compose, identity
from datashape.predicates import isscalar

from ..expr import (
    Expr, ElemWise, SimpleSelection, Sort, Apply, Distinct, Join, By, Label,
    Summary, by, ReLabel, Like, Reduction, Head
)
from .python import (
    compute, rrowfunc, rowfunc, pair_assemble, reduce_by_funcs, binops
)
from ..expr.broadcast import broadcast_collect
from ..expr.optimize import simple_selections
from ..compatibility import builtins, unicode
from ..expr import reductions
from ..dispatch import dispatch

from .core import compute_up

import py4j
from pyspark import SparkContext
import pyspark
from pyspark.rdd import RDD

__all__ = ['RDD', 'pyspark', 'SparkContext']


# PySpark adds a SIGCHLD signal handler, but that breaks other packages, so we
# remove it
# See https://issues.apache.org/jira/browse/SPARK-1394
try:
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
except:
    pass


@dispatch(Expr, RDD)
def optimize(expr, seq):
    return simple_selections(broadcast_collect(expr))


@dispatch(ElemWise, RDD)
def compute_up(t, rdd, **kwargs):
    func = rowfunc(t)
    return rdd.map(func)


@dispatch(SimpleSelection, RDD)
def compute_up(t, rdd, **kwargs):
    predicate = optimize(t.predicate, rdd)
    predicate = rrowfunc(predicate, t._child)
    return rdd.filter(predicate)


rdd_reductions = {
    reductions.sum: RDD.sum,
    reductions.min: RDD.min,
    reductions.max: RDD.max,
    reductions.count: RDD.count,
    reductions.mean: RDD.mean,
    reductions.var: RDD.variance,
    reductions.std: RDD.stdev,
    reductions.nunique: compose(RDD.count, RDD.distinct)
}


@dispatch(tuple(rdd_reductions), RDD)
def compute_up(t, rdd, **kwargs):
    return rdd_reductions[type(t)](rdd)


def istruthy(x):
    return not not x


@dispatch(reductions.any, RDD)
def compute_up(t, rdd, **kwargs):
    return istruthy(rdd.filter(identity).take(1))


@dispatch(reductions.all, RDD)
def compute_up(t, rdd, **kwargs):
    return not rdd.filter(lambda x: not x).take(1)


@dispatch(Head, RDD)
def compute_up(t, rdd, **kwargs):
    return rdd.take(t.n)


@dispatch(Apply, RDD)
def compute_up(t, rdd, **kwargs):
    if t._splittable:
        return rdd.mapPartitions(t.func)
    else:
        raise NotImplementedError("Can only apply splittable functions."
                                  "To apply function to each partition add "
                                  "splittable=True kwarg to call to apply.  "
                                  "t.apply(func, dshape, splittable=True)")


@dispatch(Sort, RDD)
def compute_up(t, rdd, **kwargs):
    if isinstance(t.key, (str, unicode, tuple, list)):
        key = rowfunc(t._child[t.key])
    else:
        key = optimize(t.key, rdd)
        key = rrowfunc(key, t._child)
    return (rdd.keyBy(key)
            .sortByKey(ascending=t.ascending)
            .map(lambda x: x[1]))


@dispatch(Distinct, RDD)
def compute_up(t, rdd, **kwargs):
    if t.on:
        raise NotImplementedError(
            'spark backend cannot specify what columns to distinct on'
        )
    return rdd.distinct()


def jgetattr(data, attr, default=None):
    """Spark's API doesn't properly implement the ``getattr`` interface, so
    we work around it.
    """
    try:
        return getattr(data, attr, default)
    except py4j.protocol.Py4JJavaError:
        return default


@compute_up.register(Join, RDD, RDD)
def spark_join(t, lhs, rhs, **kwargs):
    on_left = rowfunc(t.lhs[t.on_left])
    on_right = rowfunc(t.rhs[t.on_right])

    lhs = lhs.keyBy(on_left)
    rhs = rhs.keyBy(on_right)

    how = t.how

    if how == 'inner':
        joiner = lhs.join
    elif how == 'left':
        joiner = lhs.leftOuterJoin
    elif how == 'right':
        joiner = lhs.rightOuterJoin
    elif how == 'outer':
        joiner = lhs.fullOuterJoin
    else:
        raise ValueError("Invalid join type %r, must be one of "
                         "{'inner', 'left', 'right', 'outer'}" % how)

    rdd = joiner(rhs)
    assemble = pair_assemble(t)

    return rdd.map(lambda x: assemble(x[1]))


@dispatch(By, RDD)
def compute_up(t, rdd, **kwargs):
    grouper = optimize(t.grouper, rdd)
    apply = optimize(t.apply, rdd)
    t = by(grouper, apply)
    if ((isinstance(t.apply, Reduction) and type(t.apply) in binops) or
        (isinstance(t.apply, Summary) and
         builtins.all(type(val) in binops for val in t.apply.values))):
        grouper, binop, combiner, initial = reduce_by_funcs(t)

        if isscalar(t.grouper.dshape.measure):
            keyfunc = lambda x: (x,)
        else:
            keyfunc = identity
        if isscalar(t.apply.dshape.measure):
            valfunc = lambda x: (x,)
        else:
            valfunc = identity
        unpack = lambda kv: keyfunc(kv[0]) + valfunc(kv[1])

        create = lambda v: binop(initial, v)

        return (rdd.keyBy(grouper)
                   .combineByKey(create, binop, combiner)
                   .map(unpack))
    else:
        raise NotImplementedError("By only implemented for common reductions."
                                  "\nGot %s" % type(t.apply))


@dispatch((Label, ReLabel), RDD)
def compute_up(t, rdd, **kwargs):
    return rdd


@dispatch(Summary, RDD)
def compute_up(t, rdd, **kwargs):
    rdd = rdd.cache()
    return tuple(
        compute(value, {t._child: rdd}, return_type='native')
        for value in t.values
    )


@dispatch(Like, RDD)
def compute_up(t, rdd, **kwargs):
    def func(value, pattern=t.pattern):
        return fnmatch.fnmatch(value, pattern)
    return rdd.map(func)
