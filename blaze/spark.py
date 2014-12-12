from __future__ import absolute_import, division, print_function
from functools import partial

from into import into, convert
from .compatibility import _strtypes
from .dispatch import dispatch
from .expr import Expr
from datashape import discover, var
from collections import Iterator, Iterable

try:
    from .compute.spark import *
except (ImportError, AttributeError):
    RDD = type(None)
    SparkContext = type(None)
    pyspark = None

__all__ = ['pyspark']


@convert.register(list, RDD)
def list_to_rdd(rdd, **kwargs):
    return rdd.collect()


@dispatch(SparkContext, (Expr, object))
def into(sc, o, **kwargs):
    return sc.parallelize(convert(list, o, **kwargs))


@dispatch(RDD)
def discover(rdd):
    data = rdd.take(50)
    return var * discover(data).subshape[0]
