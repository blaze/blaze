from __future__ import absolute_import, division, print_function
from functools import partial

from .compatibility import _strtypes
from .compute.spark import *
from .compute.sparksql import *
from .sparksql import *
from .data.utils import coerce
from .dispatch import dispatch
from datashape import discover

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


@dispatch(RDD)
def discover(rdd):
    data = rdd.take(50)
    return rdd.count() * discover(data).subshape[0]
