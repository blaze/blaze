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

    # If the result is ephemeral then make it concrete
    if isinstance(result, Iterator):
        ds = expr.dshape
        result = into(concrete_type(ds), result, dshape=ds)

    # Cache result
    data.cache[expr] = result

    return result


def concrete_type(ds):
    """ A type into which we can safely deposit streaming data

    >>> concrete_type('5 * int').__name__
    'ndarray'
    >>> concrete_type('var * {name: string, amount: int}').__name__
    'DataFrame'
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
