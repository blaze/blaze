from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
from .dispatch import dispatch
from .expr import Expr, Field, ndim
from .compute import compute
from .compatibility import unicode

from odo import odo


class CachedDataset(object):
    def __init__(self, data, cache=None):
        self.data = data
        if cache is None:
            cache = dict()
        self.cache = cache


@dispatch(CachedDataset)
def discover(d, **kwargs):
    return discover(d.data, **kwargs)


@dispatch(Field, CachedDataset)
def compute_up(expr, data, **kwargs):
    return data.data[expr._name]


@dispatch(Expr, CachedDataset)
def compute_down(expr, data, **kwargs):
    if expr in data.cache:
        return data.cache[expr]

    leaf = expr._leaves()[0]

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
    <type 'float'>
    >>> concrete_type('float32')
    <type 'float'>
    >>> concrete_type('int64')
    <type 'int'>
    >>> concrete_type('int32')
    <type 'int'>
    >>> concrete_type('uint8')
    <type 'int'>
    >>> concrete_type('bool')
    <type 'bool'>
    >>> concrete_type('complex[float64]')
    <type 'complex'>
    >>> concrete_type('complex[float32]')
    <type 'complex'>
    >>> concrete_type('?int64')
    <type 'int'>
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
