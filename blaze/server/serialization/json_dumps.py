from __future__ import absolute_import, division, print_function

import datetime

import pandas as pd

from blaze.dispatch import dispatch
from datashape import Mono, DataShape
from functools import partial


json_dumps_ns = dict()
dispatch = partial(dispatch, namespace=json_dumps_ns)

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
