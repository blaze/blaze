import pandas as pd
import numpy as np
from datashape import *

def unit_to_dtype(ds):
    """

    >>> unit_to_dtype('int32')
    dtype('int32')
    >>> unit_to_dtype('float64')
    dtype('float64')
    >>> unit_to_dtype('?int64')
    dtype('float64')
    >>> unit_to_dtype('string')
    dtype('O')
    """
    if isinstance(ds, str):
        ds = dshape(ds)[0]
    if isinstance(ds, Option) and isnumeric(ds):
        return unit_to_dtype(str(ds).replace('int', 'float').replace('?', ''))
    if isinstance(ds, Option) and ds.ty in (date_, datetime_, string):
        ds = ds.ty
    return to_numpy_dtype(ds)


def dshape_to_pandas(ds):
    """

    >>> dshape_to_pandas('{a: int32}')
    ({'a': dtype('int32')}, [])

    >>> dshape_to_pandas('{a: int32, when: datetime}')
    ({'a': dtype('int32')}, ['when'])

    >>> dshape_to_pandas('{a: ?int64}')
    ({'a': dtype('float64')}, [])
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]

    dtypes = {name: unit_to_dtype(typ) for name, typ in ds.dict.items()
                if not 'date' in str(typ)}

    datetimes = [name for name, typ in ds.dict.items()
                    if 'date' in str(typ)]

    return dtypes, datetimes


def isnumeric(ds):
    """
    >>> isnumeric('int32')
    True
    >>> isnumeric('string')
    False
    >>> isnumeric('var * {amount: ?int32}')
    False
    """
    if isinstance(ds, str):
            ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    if isinstance(ds, Option):
        return isnumeric(ds.ty)
    return isinstance(ds, Unit) and np.issubdtype(to_numpy_dtype(ds), np.number)
