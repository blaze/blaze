from __future__ import absolute_import, division, print_function

from .compatibility import basestring
from .compute.mongo import dispatch

try:
    from pymongo.collection import Collection
    from pymongo import ASCENDING
except ImportError:
    Collection = type(None)


__all__ = ['create_index']


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
        if not isinstance(el, (tuple, basestring)):
            raise TypeError('indexing keys must be a string or pair of '
                            '(<column name>, <parameter>)')
        yield scrub_keys(el)


@dispatch(Collection, basestring)
def create_index(coll, key, **kwargs):
    coll.create_index(key, **kwargs)


@dispatch(Collection, list)
def create_index(coll, keys, **kwargs):
    coll.create_index(list(scrub_keys(keys)), **kwargs)
