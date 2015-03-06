from datashape import dshape, discover
from datashape.predicates import isscalar, isrecord, iscollection
import numpy as np
import pandas as pd
from .dispatch import dispatch
from .expr import Expr, Field, symbol, ndim
from .compute import compute
from .compatibility import unicode
from collections import Iterator
from odo import into


class CachedDataset(object):
    def __init__(self, data, cache=None):
        self.data = data
        if cache is None:
            cache = dict()
        self.cache = cache


@dispatch(CachedDataset)
def discover(d):
    return discover(d.data)


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
    if not iscollection(ds):
        return type(ds)
    if ndim(ds) == 1 and isrecord(ds.measure):
        return pd.DataFrame
    if ndim(ds) > 1 or isscalar(ds.measure):
        return np.ndarray
    return list
