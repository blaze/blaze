from __future__ import absolute_import, division, print_function

try:
    from pymongo.collection import Collection
except ImportError:
    Collection = type(None)

from collections import Iterator
from toolz import take, concat, partition_all
from datashape import discover, isdimension, dshape
import copy

from .dispatch import dispatch
from .data.core import DataDescriptor
from .compute.mongo import *
from .expr.table import TableExpr

__all__ = ['pymongo']

@dispatch(Collection)
def discover(coll, n=50):
    items = list(take(n, coll.find()))
    for item in items:
        del item['_id']

    ds = discover(items)

    if isdimension(ds[0]):
        return coll.count() * ds.subshape[0]
    else:
        raise ValueError("Consistent datashape not found")


