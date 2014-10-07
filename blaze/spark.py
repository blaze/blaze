from __future__ import absolute_import, division, print_function
from functools import partial

from .compatibility import _strtypes
from .compute.spark import *
from .data.utils import coerce
from .dispatch import dispatch
from .expr import Expr
from datashape import discover, var
from collections import Iterator, Iterable

__all__ = ['pyspark', 'coerce']

@dispatch(_strtypes, RDD)
def coerce(dshape, rdd):
    return rdd.mapPartitions(partial(coerce, dshape))


@dispatch(type, RDD)
def into(a, rdd, **kwargs):
    f = into.dispatch(a, type(rdd))
    return f(a, rdd, **kwargs)

@dispatch(object, RDD)
def into(o, rdd):
    return into(o, rdd.collect())


@dispatch((tuple, list, set), RDD)
def into(a, b, **kwargs):
    if not isinstance(a, type):
        a = type(a)
    b = b.collect()
    if isinstance(b[0], (tuple, list)) and not type(b[0]) == tuple:
        b = map(tuple, b)
    return a(b)


@dispatch(SparkContext, (Expr, RDD, object) + _strtypes)
def into(sc, o, **kwargs):
    return sc.parallelize(into(list, o, **kwargs))


@dispatch(RDD)
def discover(rdd):
    data = rdd.take(50)
    return var * discover(data).subshape[0]
