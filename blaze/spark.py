from functools import partial

from .compatibility import _strtypes
from .compute.spark import *
from .data.utils import coerce
from .dispatch import dispatch

@dispatch(_strtypes, RDD)
def coerce(dshape, rdd):
    return rdd.mapPartitions(partial(coerce, dshape))


@dispatch((type, object), RDD)
def into(o, rdd):
    return into(o, rdd.collect())
