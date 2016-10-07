from __future__ import absolute_import, division, print_function

import datetime
from functools import partial

from datashape import Mono, DataShape
import pandas as pd

from blaze.dispatch import dispatch
from blaze.compatibility import unicode


json_dumps_ns = dict()
dispatch = partial(dispatch, namespace=json_dumps_ns)


@dispatch(datetime.datetime)
def json_dumps(dt):
    if dt is pd.NaT:
        # NaT has an isoformat but it is totally invalid.
        # This keeps the parsing on the client side simple.
        s = u'NaT'
    else:
        s = dt.isoformat()
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        if not dt.tzname():
            s += u'Z'

    return {u'__!datetime': s}


@dispatch(frozenset)
def json_dumps(ds):
    return {u'__!frozenset': list(ds)}


@dispatch(datetime.timedelta)
def json_dumps(ds):
    return {u'__!timedelta': ds.total_seconds()}


@dispatch(Mono)
def json_dumps(m):
    return {u'__!mono': unicode(m)}


@dispatch(DataShape)
def json_dumps(ds):
    return {u'__!datashape': unicode(ds)}


@dispatch(object)
def json_dumps(ob):
    return pd.io.packers.encode(ob)
