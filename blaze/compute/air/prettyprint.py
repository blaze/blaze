# -*- coding: utf-8 -*-

"""
Pretty printing of AIR.
"""

from __future__ import print_function, division, absolute_import

import os
import re
import dis
import types

from . import pipeline

import pykit.ir


def debug_print(func, env):
    """
    Returns whether to enable debug printing.
    """
    return isinstance(func, pykit.ir.Function)

def verbose(p, func, env):
    if not debug_print(func, env):
        return pipeline.apply_transform(p, func, env)

    title = "%s [ %s %s(%s) ]" % (_passname(p), func.type.restype,
                                  _funcname(func),
                                  ", ".join(map(str, func.type.argtypes)))

    print(title.center(60).center(90, "-"))

    if isinstance(func, types.FunctionType):
        dis.dis(func)
        func, env = pipeline.apply_transform(p, func, env)
        print()
        print(func)
        return func, env

    before = _formatfunc(func)
    func, env = pipeline.apply_transform(p, func, env)
    after = _formatfunc(func)

    if before != after:
        print(pykit.ir.diff(before, after))

    return func, env

# ______________________________________________________________________

def _passname(transform):
    return transform.__name__
    #if isinstance(transform, types.ModuleType):
    #    return transform.__name__
    #else:
    #    return ".".join([transform.__module__, transform.__name__])

def _funcname(func):
    if isinstance(func, types.FunctionType):
        return func.__name__
    else:
        return func.name

def _formatfunc(func):
    if isinstance(func, types.FunctionType):
        dis.dis(func)
        return ""
    else:
        return str(func)