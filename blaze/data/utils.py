from __future__ import absolute_import, division, print_function

from dynd import nd
from collections import Iterator
from datashape import dshape, Record
from datashape.predicates import isunit, isdimension


def validate(schema, item):
    try:
        nd.array(item, dtype=str(schema))
        return True
    except:
        return False


def coerce(dshape, item):
    if isinstance(item, Iterator):
        item = list(item)
    return nd.as_py(nd.array(item, dtype=str(dshape)))


def coerce_to_ordered(ds, data):
    """ Coerce data with dicts into an ordered ND collection

    >>> from datashape import dshape

    >>> coerce_to_ordered('{x: int, y: int}', {'x': 1, 'y': 2})
    (1, 2)

    >>> coerce_to_ordered('var * {x: int, y: int}',
    ...                  [{'x': 1, 'y': 2}, {'x': 10, 'y': 20}])
    ((1, 2), (10, 20))
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds[0], Record):
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
    """
    return [rec[name] for name in schema[0].names]


def coerce_row_to_dict(schema, row):
    """

    >>> from datashape import dshape

    >>> schema = dshape('{x: int, y: int}')
    >>> coerce_row_to_dict(schema, (1, 2)) # doctest: +SKIP
    {'x': 1, 'y': 2}
    """
    return dict((name, item) for name, item in zip(schema[0].names, row))
