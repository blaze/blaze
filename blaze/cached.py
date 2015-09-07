from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd

from datashape import dshape, discover, integral, floating, boolean, complexes
from datashape.predicates import isscalar, isrecord, iscollection

from .dispatch import dispatch
from .expr import Expr, Field, ndim
from .compute import compute
from .compatibility import unicode

from odo import odo


class CachedDataset(object):
    __slots__ = 'data', 'cache'

    def __init__(self, data, cache=None):
        self.data = data
        self.cache = cache if cache is not None else {}


@dispatch(CachedDataset)
def discover(d, **kwargs):
    return discover(d.data, **kwargs)


@dispatch(Field, CachedDataset)
def compute_up(expr, data, **kwargs):
    return data.data[expr._name]


@dispatch(Expr, CachedDataset)
def compute_down(expr, data, **kwargs):
    try:
        return data.cache[expr]
    except KeyError:
        pass

    leaf, = expr._leaves()

    # Do work
    result = compute(expr, {leaf: data.data}, **kwargs)

    ds = expr.dshape
    new_type = concrete_type(ds)

    # If the result is not the concrete data type for the datashape then make
    # it concrete
    if not isinstance(result, new_type):
        result = odo(result, new_type, dshape=ds)

    # Cache result
    data.cache[expr] = result

    return result


def concrete_type(ds):
    """A type into which we can safely deposit streaming data.

    Parameters
    ----------
    ds : DataShape

    Returns
    -------
    type : type
        The concrete type corresponding to the DataShape `ds`

    Notes
    -----
    * This will return a Python type if possible
    * Option types are not handled specially. The base type of the option type
      is returned.

    Examples
    --------
    >>> concrete_type('5 * int')
    <class 'pandas.core.series.Series'>
    >>> concrete_type('var * {name: string, amount: int}')
    <class 'pandas.core.frame.DataFrame'>
    >>> concrete_type('float64')
    <... 'float'>
    >>> concrete_type('float32')
    <... 'float'>
    >>> concrete_type('int64')
    <... 'int'>
    >>> concrete_type('int32')
    <... 'int'>
    >>> concrete_type('uint8')
    <... 'int'>
    >>> concrete_type('bool')
    <... 'bool'>
    >>> concrete_type('complex[float64]')
    <... 'complex'>
    >>> concrete_type('complex[float32]')
    <... 'complex'>
    >>> concrete_type('?int64')
    <... 'int'>
    """
    if isinstance(ds, (str, unicode)):
        ds = dshape(ds)
    if not iscollection(ds) and isscalar(ds.measure):
        measure = getattr(ds.measure, 'ty', ds.measure)
        if measure in integral.types:
            return int
        elif measure in floating.types:
            return float
        elif measure in boolean.types:
            return bool
        elif measure in complexes.types:
            return complex
        else:
            return ds.measure.to_numpy_dtype().type
    if not iscollection(ds):
        return type(ds)
    if ndim(ds) == 1:
        return pd.DataFrame if isrecord(ds.measure) else pd.Series
    if ndim(ds) > 1:
        return np.ndarray
    return list
