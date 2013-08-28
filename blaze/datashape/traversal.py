# -*- coding: utf-8 -*-

"""
Type visitor that reconstructs types.
"""

from __future__ import print_function, division, absolute_import
from blaze.datashape.coretypes import Mono, CType, TypeVar, type_constructor

def transform(visitor, t):
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
        elif not isinstance(t, (TypeVar, CType)):
            tcon = type_constructor(t)
            return tcon(*[transform(visitor, p) for p in t.parameters])

    return t

def tmap(f, t):
    """
    Map f over t, calling `f` with type `t`. Reconstructs a new type with
    the results from `f`.
    """
    if isinstance(t, Mono) and not isinstance(t, (TypeVar, CType)):
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
