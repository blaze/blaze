from __future__ import absolute_import, division, print_function

from toolz import take
from datashape import discover, isdimension

from .compatibility import basestring, map
from .compute.mongo import dispatch

from pymongo.collection import Collection
from pymongo import ASCENDING


__all__ = ['discover', 'drop', 'create_index']


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


@dispatch(object)
def scrub_keys(o):
    """Add an ascending sort key when pass a string, to make the MongoDB
    interface similar to SQL.
    """
    raise NotImplementedError("scrub_keys not implemented for type %r" %
                              type(o).__name__)


@dispatch(basestring)
def scrub_keys(s):
    return s, ASCENDING


@dispatch(tuple)
def scrub_keys(t):
    return t


@dispatch(list)
def scrub_keys(seq):
    for el in seq:
        assert isinstance(el, (tuple, basestring)), \
            ('indexing keys must be a string or pair of '
             '(<column name>, <parameter>)')
        yield scrub_keys(el)


@dispatch(Collection, basestring, list)
def create_index(coll, index_name, keys, **kwargs):
    coll.create_index(list(scrub_keys(keys)), name=index_name, **kwargs)


@dispatch(Collection, basestring, basestring)
def create_index(coll, index_name, key, **kwargs):
    coll.create_index(key, name=index_name, **kwargs)
