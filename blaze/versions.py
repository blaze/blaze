from __future__ import absolute_import, division, print_function

import importlib
import itertools
import sys

from .compute import compute_up
from .compute.core import compute_down

builtins = {"__builtin__", "datetime", "_abcoll", "numbers"}

def _get_package(datatype):
    t = datatype.__module__.split(".")[0]
    if t not in builtins:
        return t
    else:
        return "__builtin__"

def get_backends():
    funcs = itertools.chain(compute_up.funcs, compute_down.funcs)
    packages = {_get_package(signature[1]) for signature in funcs if len(signature) > 1}

    versions = {}
    builtins = {"__builtin__", "datetime", "_abcoll", "numbers"}
    for package in packages:
        if package != "__builtin__":
            try:
                versions[package] = importlib.import_module(package).__version__
            except ImportError:
                versions[package] = "Not Installed"
            except AttributeError:
                versions[package] = "Unknown"
    versions["Python"] = sys.version

    return versions
