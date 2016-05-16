from __future__ import absolute_import, division, print_function

import types
import datetime
from collections import Callable

import pandas as pd

from .dispatch import dispatch
from datashape import Mono, DataShape


@dispatch(datetime.datetime)
def json_dumps(dt):
    if dt is pd.NaT:
        # NaT has an isoformat but it is totally invalid.
        # This keeps the parsing on the client side simple.
        s = 'NaT'
    else:
        s = dt.isoformat()
        if not dt.tzname():
            s += 'Z'

    return {'__!datetime': s}


@dispatch(frozenset)
def json_dumps(ds):
    return {'__!frozenset': list(ds)}


@dispatch(datetime.timedelta)
def json_dumps(ds):
    return {'__!timedelta': ds.total_seconds()}


@dispatch(Mono)
def json_dumps(m):
    return {'__!mono': str(m)}


@dispatch(DataShape)
def json_dumps(ds):
    return {'__!datashape': str(ds)}


@dispatch(types.BuiltinFunctionType)
def json_dumps(f):
    return {'__!builtin_function': f.__name__}


@dispatch(Callable)
def json_dumps(f):
    # let the server serialize any callable - this is only used for testing
    # at present - do the error handling when json comes from client so in
    # object_hook, catch anything that is not pandas_numpy
    fcn = ".".join([f.__module__, f.__name__])
    return {'__!numpy_pandas_function': fcn}
