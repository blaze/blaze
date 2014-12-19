from __future__ import absolute_import, division, print_function
from functools import reduce
from itertools import chain
import sys

import pandas as pd

from .compute import compute_up
from .compute.core import compute_down
from .compute.spark import Dummy
from .compute.mongo import MongoQuery


builtins = set(["__builtin__", "datetime", "_abcoll", "numbers", "collections",
                "builtins"])
blaze_wrappers = {MongoQuery: "pymongo", Dummy: "pyspark"}

def _get_functions():
    return chain(compute_up.funcs, compute_down.funcs)

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
    packages = set(_get_package(signature[1]) 
                   for signature in _get_functions()
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

def _get_backend(t):
    package = _get_package(t)

    if package == "__builtin__":
        return "Pure Python"
    elif package == "blaze":
        return "blaze." + t.__name__
    else:
        return package

def get_support_dict():
    support = {}
    for func in _get_functions():
        operation, types = func[0], func[1:]
        for backend in set(_get_backend(t) for t in types):
            if backend not in support:
                support[backend] = set()
            support[backend].add(operation.__name__)

    return support

def get_multiindexed_support():
    values = set(func[0] for func in _get_functions())

    ts = sorted((v.__module__.split(".")[-1], v.__name__)
                for v in values 
                if "blaze" in v.__module__) # ignore types like list or tuple
    mi = pd.MultiIndex.from_tuples(ts)
    df = pd.DataFrame(index=mi)
    operations = set(df.index.levels[-1])

    for backend, supported in get_support_dict().items():
        supported = pd.Series(dict((op, op in supported) for op in operations))
        df[backend] = supported.reindex_axis(mi, level=1)

    return df

def get_grouped_support():
    return get_multiindexed_support().groupby(level=0).any()

def get_detailed_support():
    return get_multiindexed_support().groupby(level=1).any()
