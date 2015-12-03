from __future__ import absolute_import, division, print_function

import os
import datetime
import re
from weakref import WeakKeyDictionary

try:
    from cytoolz import nth, memoize, unique, concat, first, drop
except ImportError:
    from toolz import nth, memoize, unique, concat, first, drop

from toolz.curried.operator import setitem

from itertools import islice, chain
from collections import Iterator
from multiprocessing.pool import ThreadPool

# these are used throughout blaze, don't remove them
from odo.utils import tmpfile, filetext, filetexts, raises, keywords, ignoring

import pandas as pd
import psutil
import numpy as np

# Imports that replace older utils.
from .compatibility import map, zip

from .dispatch import dispatch

thread_pool = ThreadPool(psutil.cpu_count())


def nth_list(n, seq):
    """

    >>> tuple(nth_list([0, 1, 4], 'Hello'))
    ('H', 'e', 'o')
    >>> tuple(nth_list([4, 1, 0], 'Hello'))
    ('o', 'e', 'H')
    >>> tuple(nth_list([0, 0, 0], 'Hello'))
    ('H', 'H', 'H')
    """
    seq = iter(seq)

    result = []
    old = 0
    item = next(seq)
    for index in sorted(n):
        for i in range(index - old):
            item = next(seq)
        result.append(item)
        old = index

    order = [x[1] for x in sorted(zip(n, range(len(n))))]
    return (result[i] for i in order)


def get(ind, coll, lazy=False):
    """

    >>> get(0, 'Hello')
    'H'

    >>> get([1, 0], 'Hello')
    ('e', 'H')

    >>> get(slice(1, 4), 'Hello')
    ('e', 'l', 'l')

    >>> get(slice(1, 4), 'Hello', lazy=True)
    <itertools.islice object at ...>
    """
    if isinstance(ind, list):
        result = nth_list(ind, coll)
    elif isinstance(ind, slice):
        result = islice(coll, ind.start, ind.stop, ind.step)
    else:
        if isinstance(coll, Iterator):
            result = nth(ind, coll)
        else:
            result = coll[ind]
    if not lazy and isinstance(result, Iterator):
        result = tuple(result)
    return result


def ndget(ind, data):
    """
    Get from N-Dimensional getable

    Can index with elements, lists, or slices.  Mimic's numpy fancy indexing on
    generic indexibles.

    >>> data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> ndget(0, data)
    [[1, 2], [3, 4]]
    >>> ndget((0, 1), data)
    [3, 4]
    >>> ndget((0, 0, 0), data)
    1
    >>> ndget((slice(0, 2), [0, 1], 0), data)
    ((1, 3), (5, 7))
    """
    if isinstance(ind, tuple) and len(ind) == 1:
        ind = ind[0]
    if not isinstance(ind, tuple):
        return get(ind, data)
    result = get(ind[0], data)
    if isinstance(ind[0], (list, slice)):
        return type(result)(ndget(ind[1:], row) for row in result)
    else:
        return ndget(ind[1:], result)


def normalize_to_date(dt):
    if isinstance(dt, datetime.datetime) and not dt.time():
        return dt.date()
    else:
        return dt


def assert_allclose(lhs, rhs):
    for tb in map(zip, lhs, rhs):
        for left, right in tb:
            if isinstance(left, (np.floating, float)):
                # account for nans
                assert np.all(np.isclose(left, right, equal_nan=True))
                continue
            if isinstance(left, datetime.datetime):
                left = normalize_to_date(left)
            if isinstance(right, datetime.datetime):
                right = normalize_to_date(right)
            assert left == right


def example(filename, datapath=os.path.join('examples', 'data')):
    import blaze
    return os.path.join(os.path.dirname(blaze.__file__), datapath, filename)


def available_memory():
    return psutil.virtual_memory().available


def listpack(x):
    """
    >>> listpack(1)
    [1]
    >>> listpack((1, 2))
    [1, 2]
    >>> listpack([1, 2])
    [1, 2]
    """
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    else:
        return [x]


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


def object_hook(obj):
    """Convert a json object dict back into a python object.

    This looks for our objects that have encoded richer representations with
    a ``__!{type}`` key.

    Parameters
    ----------
    obj : dict
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
    ... def _parse_my_list(obj):
    ...     return MyList(obj)

    Register can also be called as a function like:
    >>> object_hook.register('frozenset', frozenset)
    """
    if len(obj) != 1:
        return obj

    key, = obj.keys()
    if not key.startswith('__!'):
        return obj

    return object_hook._converters[key[len('__!'):]](obj[key])
object_hook._converters = {}
object_hook.register = setitem(object_hook._converters)


object_hook.register('datetime', pd.Timestamp)
object_hook.register('frozenset', frozenset)


@object_hook.register('timedelta')
def _read_timedelta(ds):
    return datetime.timedelta(seconds=ds)


def normalize(s):
    s = ' '.join(s.strip().split()).lower()
    s = re.sub(r'(alias)_?\d*', r'\1', s)
    return re.sub(r'__([A-Za-z_][A-Za-z_0-9]*)', r'\1', s)


def weakmemoize(f):
    """Memoize ``f`` with a ``WeakKeyDictionary`` to allow the arguments
    to be garbage collected.

    Parameters
    ----------
    f : callable
        The function to memoize.

    Returns
    -------
    g : callable
        ``f`` with weak memoiza
    """
    return memoize(f, cache=WeakKeyDictionary())


def ordered_intersect(*sets):
    """Set intersection of two sequences that preserves order.

    Parameters
    ----------
    sets : tuple of Sequence

    Returns
    -------
    generator

    Examples
    --------
    >>> list(ordered_intersect('abcd', 'cdef'))
    ['c', 'd']
    >>> list(ordered_intersect('bcda', 'bdfga'))
    ['b', 'd', 'a']
    >>> list(ordered_intersect('zega', 'age'))  # 1st sequence determines order
    ['e', 'g', 'a']
    >>> list(ordered_intersect('gah', 'bag', 'carge'))
    ['g', 'a']
    """
    common = frozenset.intersection(*map(frozenset, sets))
    return (x for x in unique(concat(sets)) if x in common)
