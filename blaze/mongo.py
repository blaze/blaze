from __future__ import absolute_import, division, print_function

try:
    from pymongo.collection import Collection
except ImportError:
    Collection = type(None)

from toolz import take
from datashape import discover, isdimension

from .dispatch import dispatch
from .compute.mongo import *

__all__ = ['discover', 'drop']


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


@dispatch(Collection)
def drop(m):
    m.drop()
