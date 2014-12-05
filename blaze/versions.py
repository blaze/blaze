from __future__ import absolute_import, division, print_function

import itertools
import sys

from .compute import compute_up
from .compute.core import compute_down
from .compute.spark import Dummy
from .compute.mongo import MongoQuery


builtins = set(["__builtin__", "datetime", "_abcoll", "numbers", "collections",
                "builtins"])
blaze_wrappers = {MongoQuery: "pymongo", Dummy: "pyspark"}

def _get_package(datatype):
    if datatype in blaze_wrappers:
        return blaze_wrappers[datatype]
    else:
        t = datatype.__module__.split(".")[0]
        if t not in builtins:
            return t
        else:
            return "__builtin__"

def get_backends():
    funcs = itertools.chain(compute_up.funcs, compute_down.funcs)
    packages = set(_get_package(signature[1]) 
                   for signature in funcs 
                   if len(signature) > 1)

    versions = {}
    for package in packages:
        if package != "__builtin__":
            try:
                p = __import__(package)
                if hasattr(p, "__version__"):
                    versions[package] = p.__version__
                elif hasattr(p, "version"):
                    versions[package] = p.version
                else:
                    versions[package] = "Unknown"
            except ImportError:
                versions[package] = "Not Installed"
    versions["Python"] = sys.version

    return versions

def _get_support(t):
    package = _get_package(t)

    if package == "__builtin__":
        if t in frozenset([list, tuple, dict, float, int, bool]):
            return "Pure Python"
        else:
            return t
    elif package == "blaze":
        return t
    else:
        return package

def get_compute_support():
    funcs = itertools.chain(compute_up.ordering, compute_down.ordering)
    support = {}

    for func in funcs:
        operation, types = func[0], func[1:]
        supported = set(_get_support(t) for t in types)
        if types:
            if operation not in support:
                support[operation] = supported
            else:
                support[operation] |= supported
    return support
