# -*- coding: utf-8 -*-

"""
Type visitor that reconstructs types.
"""

from __future__ import print_function, division, absolute_import
from blaze.datashape.coretypes import Mono, CType, TypeVar, type_constructor

def descend(t):
    """Determine whether to descend down the given term (which is a type)"""
    return isinstance(t, Mono) and not isinstance(t, (TypeVar, CType))

def transform(visitor, t, descend=descend):
    """
    Transform a type to recreate a new type
    """
    # def f(ds):
    #     fn = getattr(visitor, type(ds).__name__, None)
    #     if fn is not None:
    #         return fn(ds)
    #     return ds
    #
    # return tmap(f, t)

    if isinstance(t, Mono):
        fn = getattr(visitor, type(t).__name__, None)
        if fn is not None:
            return fn(t)
        elif descend(t):
            tcon = type_constructor(t)
            return tcon(*[transform(visitor, p) for p in t.parameters])

    return t

def tmap(f, t, descend=descend):
    """
    Map f over t, calling `f` with type `t`. Reconstructs a new type with
    the results from `f`.
    """
    if descend(t):
        tcon = type_constructor(t)
        t = tcon(*[tmap(f, p) for p in t.parameters])
    return f(t)

def traverse(f, t):
    """
    Map f over t, calling `f` with type `t` and the map result of the mapping
    `f` over `t`s parameters.
    """
    if isinstance(t, Mono):
        return f(t, [traverse(f, p) for p in t.parameters])
    return t
