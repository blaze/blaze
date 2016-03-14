from __future__ import absolute_import, division, print_function

from weakref import WeakKeyDictionary

from toolz import curry, concat, first, memoize
from multipledispatch import MDNotImplementedError

from ..expr import (
    Distinct,
    ElemWise,
    Expr,
    Field,
    Head,
    Projection,
    Slice,
    Symbol,
    path,
    symbol,
)
from ..expr.optimize import lean_projection, simple_selections
from ..expr.split import split
from ..partition import partitions
from .core import compute
from .pmap import get_default_pmap

from collections import Iterator, Iterable
import datashape
import bcolz
import numpy as np
import pandas as pd


from ..dispatch import dispatch
from odo import into

__all__ = ['bcolz']

COMFORTABLE_MEMORY_SIZE = 1e9


@memoize(cache=WeakKeyDictionary())
def box(type_):
    """Create a non-iterable box type for an object.

    Parameters
    ----------
    type_ : type
        The type to create a box for.

    Returns
    -------
    box : type
        A type to box values of type ``type_``.
    """
    class c(object):
        __slots__ = 'value',

        def __init__(self, value):
            if not isinstance(value, type_):
                raise TypeError(
                    "values must be of type '%s' (received '%s')" % (
                        type_.__name__, type(value).__name__,
                    ),
                )
            self.value = value

    c.__name__ = 'box(%s)' % type_.__name__
    return c


@dispatch(Expr, (box(bcolz.ctable), box(bcolz.carray)))
def optimize(expr, _):
    return simple_selections(lean_projection(expr))


@dispatch(Expr, (bcolz.ctable, bcolz.carray))
def pre_compute(expr, data, scope=None, **kwargs):
    # box the data so that we don't need to deal with ambiguity of ctable
    # and carray being instances of the Iterator ABC.
    return box(type(data))(data)


@dispatch(Expr, (box(bcolz.ctable), box(bcolz.carray)))
def post_compute(expr, data, **kwargs):
    # Unbox the bcolz objects.
    return data.value


@dispatch((box(bcolz.carray), box(bcolz.ctable)))
def discover(data):
    val = data.value
    return datashape.from_numpy(val.shape, val.dtype)

Cheap = (Head, ElemWise, Distinct, Symbol)


@dispatch(Head, (box(bcolz.ctable), box(bcolz.carray)))
def compute_down(expr, data, **kwargs):
    """ Cheap and simple computation in simple case

    If we're given a head and the entire expression is cheap to do (e.g.
    elemwises, selections, ...) then compute on data directly, without
    parallelism"""
    leaf = expr._leaves()[0]
    if all(isinstance(e, Cheap) for e in path(expr, leaf)):
        val = data.value
        return compute(
            expr,
            {leaf: into(Iterator, val)},
            return_type='native',
            **kwargs
        )
    else:
        raise MDNotImplementedError()


@dispatch(Field, box(bcolz.ctable))
def compute_up(expr, data, **kwargs):
    return data.value[str(expr._name)]


@dispatch(Projection, box(bcolz.ctable))
def compute_up(expr, data, **kwargs):
    return data.value[list(map(str, expr.fields))]


@dispatch(Slice, (box(bcolz.carray), box(bcolz.ctable)))
def compute_up(expr, data, **kwargs):
    return data.value[expr.index]


def compute_chunk(source, chunk, chunk_expr, data_index):
    part = source[data_index]
    return compute(chunk_expr, {chunk: part}, return_type='native')


def get_chunksize(data):
    if isinstance(data, bcolz.carray):
        return data.chunklen
    elif isinstance(data, bcolz.ctable):
        return min(data[c].chunklen for c in data.names)
    else:
        raise TypeError("Don't know how to compute chunksize for type %r" %
                        type(data).__name__)


@dispatch(Expr, (box(bcolz.carray), box(bcolz.ctable)))
def compute_down(expr, data, chunksize=None, map=None, **kwargs):
    data = data.value
    if map is None:
        map = get_default_pmap()

    leaf = expr._leaves()[0]

    if chunksize is None:
        chunksize = max(2**16, get_chunksize(data))

    # If the bottom expression is a projection or field then want to do
    # compute_up first
    children = {
        e for e in expr._traverse()
        if isinstance(e, Expr)
        and any(i is expr._leaves()[0] for i in e._inputs)
    }
    if len(children) == 1 and isinstance(first(children), (Field, Projection)):
        raise MDNotImplementedError()

    chunk = symbol('chunk', chunksize * leaf.schema)
    (chunk, chunk_expr), (agg, agg_expr) = split(leaf, expr, chunk=chunk)

    data_parts = partitions(data, chunksize=(chunksize,))

    parts = list(map(curry(compute_chunk, data, chunk, chunk_expr),
                     data_parts))

    if isinstance(parts[0], np.ndarray):
        intermediate = np.concatenate(parts)
    elif isinstance(parts[0], pd.DataFrame):
        intermediate = pd.concat(parts)
    elif isinstance(parts[0], Iterable):
        intermediate = list(concat(parts))
    else:
        raise TypeError("Don't know how to concatenate objects of type %r" %
                        type(parts[0]).__name__)

    return compute(agg_expr, {agg: intermediate}, return_type='native')


def _asarray(a):
    if isinstance(a, (bcolz.carray, bcolz.ctable)):
        return a[:]
    return np.array(list(a))


@compute_down.register(Expr, (box(bcolz.carray), box(bcolz.ctable)), Iterable)
@compute_down.register(Expr, Iterable, (box(bcolz.carray), box(bcolz.ctable)))
def bcolz_mixed(expr, a, b, **kwargs):
    return compute(
        expr,
        dict(zip(expr._leaves(), map(_asarray, (a.value, b.value)))),
        return_type='native',
    )
