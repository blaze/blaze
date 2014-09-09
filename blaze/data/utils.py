from __future__ import absolute_import, division, print_function

from itertools import chain
from dynd import nd
from collections import Iterator
from datashape import dshape, Record, DataShape
from datashape.predicates import isdimension
from toolz import partition_all, partial, map
from ..dispatch import dispatch

from ..compatibility import _strtypes


def validate(schema, item):
    try:
        nd.array(item, dtype=str(schema))
        return True
    except:
        return False


@dispatch(DataShape, object)
def coerce(dshape, item):
    return coerce(str(dshape), item)


@dispatch(_strtypes, object)
def coerce(dshape, item):
    return nd.as_py(nd.array(item, dtype=dshape), tuple=True)


@dispatch(_strtypes, Iterator)
def coerce(dshape, item):
    blocks = partition_all(1024, item)
    return chain.from_iterable(map(partial(coerce, dshape), blocks))


def coerce_to_ordered(ds, data):
    """ Coerce data with dicts into an ordered ND collection

    >>> from datashape import dshape

    >>> coerce_to_ordered('{x: int, y: int}', {'x': 1, 'y': 2})
    (1, 2)

    >>> coerce_to_ordered('var * {x: int, y: int}',
    ...                  [{'x': 1, 'y': 2}, {'x': 10, 'y': 20}])
    ((1, 2), (10, 20))

    Idempotent
    >>> coerce_to_ordered('var * {x: int, y: int}',
    ...                   ((1, 2), (10, 20)))
    ((1, 2), (10, 20))
    """
    if isinstance(ds, _strtypes):
        ds = dshape(ds)
    if isinstance(ds[0], Record):
        if isinstance(data, (list, tuple)):
            return data
        rec = ds[0]
        return tuple(coerce_to_ordered(rec[name], data[name])
                     for name in rec.names)
    if isdimension(ds[0]):
        return tuple(coerce_to_ordered(ds.subarray(1), row)
                     for row in data)
    return data


def coerce_record_to_row(schema, rec):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_record_to_row(schema, {'x': 1, 'y': 2})
    [1, 2]

    Idempotent
    >>> coerce_record_to_row(schema, [1, 2])
    [1, 2]
    """
    if isinstance(rec, (tuple, list)):
        return rec
    return [rec[name] for name in schema[0].names]


def coerce_row_to_dict(schema, row):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_row_to_dict(schema, (1, 2)) # doctest: +SKIP
    {'x': 1, 'y': 2}

    Idempotent
    >>> coerce_row_to_dict(schema, {'x': 1, 'y': 2}) # doctest: +SKIP
    {'x': 1, 'y': 2}
    """
    if isinstance(row, dict):
        return row
    return dict((name, item) for name, item in zip(schema[0].names, row))


def ordered_index(ind, ds):
    """ Transform a named index into an ordered one

    >>> ordered_index(1, '3 * int')
    1
    >>> ordered_index('name', '{name: string, amount: int}')
    0
    >>> ordered_index((0, 0), '3 * {x: int, y: int}')
    (0, 0)
    >>> ordered_index([0, 1], '3 * {x: int, y: int}')
    [0, 1]
    >>> ordered_index(([0, 1], 'x'), '3 * {x: int, y: int}')
    ([0, 1], 0)
    >>> ordered_index((0, 'x'), '3 * {x: int, y: int}')
    (0, 0)
    >>> ordered_index((0, [0, 1]), '3 * {x: int, y: int}')
    (0, [0, 1])
    >>> ordered_index((0, ['x', 'y']), '3 * {x: int, y: int}')
    (0, [0, 1])
    """
    if isinstance(ds, _strtypes):
        ds = dshape(ds)
    if isinstance(ind, (int, slice)):
        return ind
    if isinstance(ind, list):
        return [ordered_index(i, ds) for i in ind]
    if isinstance(ind, _strtypes) and isinstance(ds[0], Record):
        return ds[0].names.index(ind)
    if isinstance(ind, tuple) and not ind:
        return ()
    if isdimension(ds[0]):
        return (ind[0],) + tupleit(ordered_index(ind[1:], ds.subshape[0]))
    if isinstance(ind, tuple):
        return ((ordered_index(ind[0], ds),)
                + tupleit(ordered_index(ind[1:], ds.subshape[0])))
    raise NotImplementedError("Rule for ind: %s, ds: %ds not found"
                              % (str(ind), str(ds)))


def tupleit(x):
    if not isinstance(x, tuple):
        return (x,)
    else:
        return x


def tuplify(x):
    if isinstance(x, (tuple, list, Iterator)):
        return tuple(map(tuplify, x))
    else:
        return x
