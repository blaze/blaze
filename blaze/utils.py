from __future__ import absolute_import, division, print_function

import datetime
from collections import Iterator
from itertools import islice, product
import os
import re

try:
    from cytoolz import nth, unique, concat
except ImportError:
    from toolz import nth, unique, concat

import numpy as np
# these are used throughout blaze, don't remove them
from odo.utils import tmpfile, filetext, filetexts, raises, keywords, ignoring
import pandas as pd
import psutil
import sqlalchemy as sa

# Imports that replace older utils.
from .compatibility import map, zip


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


def normalize(s):
    """Normalize a sql expression for comparison in tests.

    Parameters
    ----------
    s : str or Selectable
        The expression to normalize. If this is a selectable, it will be
        compiled with literals inlined.

    Returns
    -------
    cs : Any
        An object that can be compared against another normalized sql
        expression.
    """
    if isinstance(s, sa.sql.Selectable):
        s = literal_compile(s)
    s = re.sub(r'(\(|\))', r' \1 ', s)       # normalize spaces around parens
    s = ' '.join(s.strip().split()).lower()  # normalize whitespace and case
    s = re.sub(r'(alias)_?\d*', r'\1', s)    # normalize aliases
    return re.sub(r'__([A-Za-z_][A-Za-z_0-9]*)', r'\1', s)


def literal_compile(s):
    """Compile a sql expression with bind params inlined as literals.

    Parameters
    ----------
    s : Selectable
        The expression to compile.

    Returns
    -------
    cs : str
        An equivalent sql string.
    """
    return str(s.compile(compile_kwargs={'literal_binds': True}))


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


class attribute(object):
    """An attribute that can be overridden by instances.
    This is like a non data descriptor property.

    Parameters
    ----------
    f : callable
        The function to execute.
    """
    def __init__(self, f):
        self._f = f

    def __get__(self, instance, owner):
        if instance is None:
            return self

        return self._f(instance)


def parameter_space(*args):
    """Unpack a sequence of positional parameter spaces into the product of each
    space.

    Parameters
    ----------
    *args
        The parameters spaces to create a product of.

    Returns
    -------
    param_space : tuple[tuple]
        The product of each of the spaces.

    Examples
    --------
    # trivial case
    >>> parameter_space(0, 1, 2)
    ((0, 1, 2),)

    # two 2-tuples
    >>> parameter_space((0, 1), (2, 3))
    ((0, 2), (0, 3), (1, 2), (1, 3))

    Notes
    -----
    This is a convenience for passing to :func:`pytest.mark.parameterized`
    """
    return tuple(product(*(
        arg if isinstance(arg, tuple) else (arg,) for arg in args
    )))
