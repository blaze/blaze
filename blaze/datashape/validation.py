# -*- coding: utf-8 -*-

"""
Datashape validation.
"""

from blaze import error
from blaze.datashape import coretypes as T

def validate(ds):
    """
    Validate a Blaze type to see whether it is well-formed.

        >>> import blaze
        >>> validate(blaze.dshape('10, int32'))
        >>> validate(blaze.dshape('*, int32'))
        >>> validate(blaze.dshape('*, *, int32'))
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
        >>> validate(blaze.dshape('T, *, T2, *, int32 -> T, X'))
        Traceback (most recent call last):
            ...
        DataShapeError: Can only use a single wildcard
    """
    T.traverse(_validate, ds)

def _validate(ds, params):
    if isinstance(ds, T.DataShape):
        wildcards = [x for x in ds.parameters if isinstance(x, T.Wild)]
        if len(wildcards) > 1:
            raise error.DataShapeError("Can only use a single wildcard")

        for x in ds.parameters[:-1]:
            if isinstance(x, T.Implements):
                # TODO: What about further constaints on the dimensions?
                raise error.DataShapeError(
                    "Only the measure can have constraints")


if __name__ == '__main__':
    import doctest
    doctest.testmod()