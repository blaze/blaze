# -*- coding: utf-8 -*-

"""
Type visitor that reconstructs types.
"""

from __future__ import print_function, division, absolute_import
from blaze.datashape.coretypes import Mono, CType, type_constructor

def transform(visitor, ds):
    """
    Transform a type to recreate a new type
    """
    if isinstance(ds, Mono):
        fn = getattr(visitor, type(ds).__name__, None)
        if fn is not None:
            return fn(ds)
        elif not isinstance(ds, CType):
            tcon = type_constructor(ds)
            return tcon(*[transform(visitor, p) for p in ds.parameters])

    return ds

def traverse(f, t):
    """
    Map f over t, calling `f` with type `t` and the map result of the mapping
    `f` over `t`s parameters.
    """
    if isinstance(t, Mono):
        return f(t, [traverse(f, p) for p in t.parameters])
    return t
