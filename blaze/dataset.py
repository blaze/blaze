from datashape import dshape, discover
from datashape.predicates import isscalar, isrecord, iscollection
import numpy as np
import pandas as pd
from .dispatch import dispatch
from .expr import Expr, Field, symbol, ndim
from .compute import compute
from collections import Iterator
from into import into


class Dataset(object):
    def __init__(self, ns, cache=None):
        self.ns = ns
        if cache is None:
            cache = dict()
        self.cache = cache


@dispatch(Dataset)
def discover(d):
    return discover(d.ns)


@dispatch(Field, Dataset)
def compute_up(expr, data, **kwargs):
    return data.ns[expr._name]


@dispatch(Expr, Dataset)
def compute_down(expr, data, **kwargs):
    if expr in data.cache:
        return data.cache[expr]

    # Replace expression based on dataset to expression based on leaves in
    # dataset's namespace
    # Also, create appropriate namespace
    leaf = expr._leaves()[0]
    leaves = dict((k, symbol(k, discover(v))) for k, v in data.ns.items())
    expr2 = expr._subs(dict((leaf[k], leaves[k]) for k in leaves))
    ns = dict((leaves[k], v) for k, v in data.ns.items())

    # Do work
    result = compute(expr2, ns, **kwargs)

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
    if isinstance(ds, str):
        ds = dshape(ds)
    if not iscollection(ds):
        return type(ds)
    if ndim(ds) == 1 and isrecord(ds.measure):
        return pd.DataFrame
    if ndim(ds) > 1 or isscalar(ds.measure):
        return np.ndarray
    return list
