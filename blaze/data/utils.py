from __future__ import absolute_import, division, print_function

from dynd import nd


def validate(schema, item):
    try:
        nd.array(item, dtype=str(schema))
        return True
    except:
        return False


def coerce(schema, item):
    return nd.as_py(nd.array(item, dtype=str(schema)))


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
