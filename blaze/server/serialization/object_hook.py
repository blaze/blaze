from __future__ import absolute_import, division, print_function

import datetime

try:
    from cytoolz import curry
except ImportError:
    from toolz import curry

from datashape import dshape
import numpy as np
import pandas as pd
from pandas.io.packers import decode

# Imports that replace older utils.
from blaze.compatibility import PY2

# dict of converters. This is stored as a default arg to object hook for
# performance because this function is really really slow when unpacking data.
# This is a mutable default but it is not a bug!
_converters = {}


if PY2:
    _keys = dict.keys
else:
    def _keys(d, _dkeys=dict.keys, _list=list):
        return _list(_dkeys(d))


def object_hook(ob,
                # Cached for performance. Forget these exist.
                _len=len,
                _keys=_keys,
                _first_three_chars=np.s_[:3],
                _converters=_converters):
    """Convert a json object dict back into a python object.

    This looks for our objects that have encoded richer representations
    with a ``__!{type}`` key.

    Parameters
    ----------
    ob : dict
        The raw json parsed dictionary.

    Returns
    -------
    parsed : any
        The richer form of the object.

    Notes
    -----
    The types that this reads can be extended with the ``register`` method.
    For example:

    >>> class MyList(list):
    ...     pass
    >>> @object_hook.register('MyList')
    ... def _parse_my_list(ob):
    ...     return MyList(ob)

    Register can also be called as a function like:
    >>> a = object_hook.register('frozenset', frozenset)
    >>> a is frozenset
    True
    """
    if _len(ob) != 1:
        return decode(ob)

    key = _keys(ob)[0]
    if key[_first_three_chars] != '__!':
        return ob

    return _converters[key](ob[key])


@curry
def register(typename, converter, converters=_converters):
    converters['__!' + typename] = converter
    return converter
object_hook.register = register

object_hook._converters = _converters  # make this accesible for debugging
del _converters
del _keys  # captured by default args


object_hook.register('datetime', pd.Timestamp)
object_hook.register('frozenset', frozenset)
object_hook.register('datashape', dshape)


@object_hook.register('mono')
def _read_mono(m):
    return dshape(m).measure


@object_hook.register('timedelta')
def _read_timedelta(ds):
    return datetime.timedelta(seconds=ds)


@object_hook.register('bytes')
def _read_bytes(bs):
    if not isinstance(bs, bytes):
        bs = bs.encode('latin1')
    return bs
