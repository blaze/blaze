from __future__ import absolute_import, division, print_function

import datetime

import numpy as np
from pandas import DataFrame, Series
from datashape import to_numpy, to_numpy_dtype
from numbers import Number

from ..expr import (
    Reduction, Field, Projection, Broadcast, Selection, ndim,
    Distinct, Sort, Tail, Head, Label, ReLabel, Expr, Slice, Join,
    std, var, count, nunique, Summary, IsIn,
    BinOp, UnaryOp, USub, Not, nelements, Repeat, Concat, Interp,
    UTCFromTimestamp, DateTimeTruncate,
    Transpose, TensorDot, Coerce, isnan,
    greatest, least, BinaryMath, atan2, Coalesce, Cast
)
from ..utils import keywords

from .core import base, compute
from .pandas import array_coalesce
from ..dispatch import dispatch
from odo import into
import pandas as pd

__all__ = ['np']


@dispatch(Field, np.ndarray)
def compute_up(c, x, **kwargs):
    if x.dtype.names and c._name in x.dtype.names:
        return x[c._name]
    if not x.dtype.names and x.shape[1] == len(c._child.fields):
        return x[:, c._child.fields.index(c._name)]
    raise NotImplementedError()  # pragma: no cover


@dispatch(Projection, np.ndarray)
def compute_up(t, x, **kwargs):
    if x.dtype.names and all(col in x.dtype.names for col in t.fields):
        return x[t.fields]
    if not x.dtype.names and x.shape[1] == len(t._child.fields):
        return x[:, [t._child.fields.index(col) for col in t.fields]]
    raise NotImplementedError()  # pragma: no cover


try:
    from .numba import broadcast_numba as broadcast_ndarray
except ImportError:
    def broadcast_ndarray(t, *data, **kwargs):
        del kwargs['scope']
        d = dict(zip(t._scalar_expr._leaves(), data))
        return compute(t._scalar_expr, d, return_type='native', **kwargs)


compute_up.register(Broadcast, np.ndarray)(broadcast_ndarray)
for i in range(2, 6):
    compute_up.register(Broadcast, *([(np.ndarray, Number)] * i))(broadcast_ndarray)


@dispatch(Repeat, np.ndarray)
def compute_up(t, data, _char_mul=np.char.multiply, **kwargs):
    if isinstance(t.lhs, Expr):
        return _char_mul(data, t.rhs)
    else:
        return _char_mul(t.lhs, data)


@compute_up.register(Repeat, np.ndarray, (np.ndarray, base))
@compute_up.register(Repeat, base, np.ndarray)
def compute_up_np_repeat(t, lhs, rhs, _char_mul=np.char.multiply, **kwargs):
    return _char_mul(lhs, rhs)


def _interp(arr, v, _Series=pd.Series, _charmod=np.char.mod):
    """
    Delegate to the most efficient string formatting technique based on
    the length of the array.
    """
    if len(arr) >= 145:
        return _Series(arr) % v

    return _charmod(arr, v)


@dispatch(Interp, np.ndarray)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return _interp(data, t.rhs)
    else:
        return _interp(t.lhs, data)


@compute_up.register(Interp, np.ndarray, (np.ndarray, base))
@compute_up.register(Interp, base, np.ndarray)
def compute_up_np_interp(t, lhs, rhs, **kwargs):
    return _interp(lhs, rhs)


@compute_up.register(greatest, np.ndarray, (np.ndarray, base))
@compute_up.register(greatest, base, np.ndarray)
def compute_up_greatest(expr, lhs, rhs, **kwargs):
    return np.maximum(lhs, rhs)


@compute_up.register(least, np.ndarray, (np.ndarray, base))
@compute_up.register(least, base, np.ndarray)
def compute_up_least(expr, lhs, rhs, **kwargs):
    return np.minimum(lhs, rhs)


@dispatch(BinOp, np.ndarray, (np.ndarray, base))
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, base, np.ndarray)
def compute_up(t, lhs, rhs, **kwargs):
    return t.op(lhs, rhs)


@dispatch(BinOp, np.ndarray)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return t.op(data, t.rhs)
    else:
        return t.op(t.lhs, data)


@compute_up.register(BinaryMath, np.ndarray, (np.ndarray, base))
@compute_up.register(BinaryMath, base, np.ndarray)
def compute_up_binary_math(t, lhs, rhs, **kwargs):
    return getattr(np, type(t).__name__)(lhs, rhs)


@dispatch(BinaryMath, np.ndarray)
def compute_up(t, data, **kwargs):
    func = getattr(np, type(t).__name__)
    if isinstance(t.lhs, Expr):
        return func(data, t.rhs)
    else:
        return func(t.lhs, data)


@compute_up.register(atan2, np.ndarray, (np.ndarray, base))
@compute_up.register(atan2, base, np.ndarray)
def compute_up_binary_math(t, lhs, rhs, **kwargs):
    return np.arctan2(lhs, rhs)


@dispatch(atan2, np.ndarray)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        return np.arctan2(data, t.rhs)
    else:
        return np.arctan2(t.lhs, data)


@dispatch(UnaryOp, np.ndarray)
def compute_up(t, x, **kwargs):
    return getattr(np, t.symbol)(x)


@dispatch(Not, np.ndarray)
def compute_up(t, x, **kwargs):
    return np.logical_not(x)


@dispatch(USub, np.ndarray)
def compute_up(t, x, **kwargs):
    return np.negative(x)


inat = np.datetime64('NaT').view('int64')


@dispatch(count, np.ndarray)
def compute_up(t, x, **kwargs):
    result_dtype = to_numpy_dtype(t.dshape)
    if issubclass(x.dtype.type, (np.floating, np.object_)):
        return pd.notnull(x).sum(keepdims=t.keepdims, axis=t.axis,
                                 dtype=result_dtype)
    elif issubclass(x.dtype.type, np.datetime64):
        return (x.view('int64') != inat).sum(keepdims=t.keepdims, axis=t.axis,
                                             dtype=result_dtype)
    else:
        return np.ones(x.shape, dtype=result_dtype).sum(keepdims=t.keepdims,
                                                        axis=t.axis,
                                                        dtype=result_dtype)


@dispatch(nunique, np.ndarray)
def compute_up(t, x, **kwargs):
    assert t.axis == tuple(range(ndim(t._child)))
    result = len(np.unique(x))
    if t.keepdims:
        result = np.array([result])
    return result


@dispatch(Reduction, np.ndarray)
def compute_up(t, x, **kwargs):
    # can't use the method here, as they aren't Python functions
    reducer = getattr(np, t.symbol)
    if 'dtype' in keywords(reducer):
        return reducer(x, axis=t.axis, keepdims=t.keepdims,
                       dtype=to_numpy_dtype(t.schema))
    return reducer(x, axis=t.axis, keepdims=t.keepdims)


def axify(expr, axis, keepdims=False):
    """ inject axis argument into expression

    Helper function for compute_up(Summary, np.ndarray)

    >>> from blaze import symbol
    >>> s = symbol('s', '10 * 10 * int')
    >>> expr = s.sum()
    >>> axify(expr, axis=0)
    sum(s, axis=(0,))
    """
    return type(expr)(expr._child, axis=axis, keepdims=keepdims)


@dispatch(Summary, np.ndarray)
def compute_up(expr, data, **kwargs):
    shape, dtype = to_numpy(expr.dshape)
    if shape:
        result = np.empty(shape=shape, dtype=dtype)
        for n, v in zip(expr.names, expr.values):
            result[n] = compute(
                axify(v, expr.axis, expr.keepdims),
                data,
                return_type='native',
            )
        return result
    else:
        return tuple(
            compute(axify(v, expr.axis), data, return_type='native')
            for v in expr.values
        )


@dispatch((std, var), np.ndarray)
def compute_up(t, x, **kwargs):
    return getattr(x, t.symbol)(ddof=t.unbiased, axis=t.axis,
            keepdims=t.keepdims)


@compute_up.register(Distinct, np.recarray)
def recarray_distinct(t, rec, **kwargs):
    return pd.DataFrame.from_records(rec).drop_duplicates(
        subset=t.on or None).to_records(index=False).astype(rec.dtype)


@dispatch(Distinct, np.ndarray)
def compute_up(t, arr, _recarray_distinct=recarray_distinct, **kwargs):
    if t.on:
        if getattr(arr.dtype, 'names', None) is not None:
            return _recarray_distinct(t, arr, **kwargs).view(np.ndarray)
        else:
            raise ValueError('malformed expression: no columns to distinct on')

    return np.unique(arr)


@dispatch(Sort, np.ndarray)
def compute_up(t, x, **kwargs):
    if x.dtype.names is None:  # not a struct array
        result = np.sort(x)
    elif (t.key in x.dtype.names or  # struct array
        isinstance(t.key, list) and all(k in x.dtype.names for k in t.key)):
        result = np.sort(x, order=t.key)
    elif t.key:
        raise NotImplementedError("Sort key %s not supported" % t.key)

    if not t.ascending:
        result = result[::-1]

    return result


@dispatch(Head, np.ndarray)
def compute_up(t, x, **kwargs):
    return x[:t.n]


@dispatch(Tail, np.ndarray)
def compute_up(t, x, **kwargs):
    return x[-t.n:]


@dispatch(Label, np.ndarray)
def compute_up(t, x, **kwargs):
    return np.array(x, dtype=[(t.label, x.dtype.type)])


@dispatch(ReLabel, np.ndarray)
def compute_up(t, x, **kwargs):
    types = [x.dtype[i] for i in range(len(x.dtype))]
    return np.array(x, dtype=list(zip(t.fields, types)))


@dispatch(Selection, np.ndarray)
def compute_up(sel, x, **kwargs):
    predicate = compute(sel.predicate, {sel._child: x}, return_type='native')
    cond = getattr(predicate, 'values', predicate)
    return x[cond]


@dispatch(Selection, np.ndarray, np.ndarray)
def compute_up(expr, arr, predicate, **kwargs):
    return arr[predicate]


@dispatch(Selection, np.ndarray, Series)
def compute_up(expr, arr, predicate, **kwargs):
    return arr[predicate.values]


@dispatch(UTCFromTimestamp, np.ndarray)
def compute_up(expr, data, **kwargs):
    return (data * 1e6).astype('datetime64[us]')


@dispatch(Slice, np.ndarray)
def compute_up(expr, x, **kwargs):
    return x[expr.index]


@dispatch(Cast, np.ndarray)
def compute_up(t, x, **kwargs):
    # resolve ambiguity
    return x


@dispatch(Expr, np.ndarray)
def compute_up(t, x, **kwargs):
    ds = t._child.dshape
    if x.ndim > 1 or isinstance(x, np.recarray) or x.dtype.fields is not None:
        return compute_up(t, into(DataFrame, x, dshape=ds), **kwargs)
    else:
        return compute_up(t, into(Series, x, dshape=ds), **kwargs)


@dispatch(nelements, np.ndarray)
def compute_up(expr, data, **kwargs):
    axis = expr.axis
    if expr.keepdims:
        shape = tuple(data.shape[i] if i not in axis else 1
                                    for i in range(ndim(expr._child)))
    else:
        shape = tuple(data.shape[i] for i in range(ndim(expr._child))
                      if i not in axis)
    value = np.prod([data.shape[i] for i in axis])
    result = np.empty(shape)
    result.fill(value)
    result = result.astype('int64')

    return result



# Note the use of 'week': 'M8[D]' here.

# We truncate week offsets "manually" in the compute_up implementation by first
# converting to days then multiplying our measure by 7 this simplifies our code
# by only requiring us to calculate the week offset relative to the day of week.

precision_map = {'year': 'M8[Y]',
                 'month': 'M8[M]',
                 'week': 'M8[D]',
                 'day': 'M8[D]',
                 'hour': 'M8[h]',
                 'minute': 'M8[m]',
                 'second': 'M8[s]',
                 'millisecond': 'M8[ms]',
                 'microsecond': 'M8[us]',
                 'nanosecond': 'M8[ns]'}


# these offsets are integers in units of their representation

epoch = datetime.datetime(1970, 1, 1)
offsets = {
    'week': epoch.isoweekday(),
    'day': epoch.toordinal() # number of days since *Python's* epoch (01/01/01)
}


@dispatch(DateTimeTruncate, (np.ndarray, np.datetime64))
def compute_up(expr, data, **kwargs):
    np_dtype = precision_map[expr.unit]
    offset = offsets.get(expr.unit, 0)
    measure = expr.measure * 7 if expr.unit == 'week' else expr.measure
    result = (((data.astype(np_dtype)
                    .view('int64')
                    + offset)
                    // measure
                    * measure
                    - offset)
                    .astype(np_dtype))
    return result


@dispatch(isnan, np.ndarray)
def compute_up(expr, data, **kwargs):
    return np.isnan(data)


@dispatch(np.ndarray)
def chunks(x, chunksize=1024):
    start = 0
    n = len(x)
    while start < n:
        yield x[start:start + chunksize]
        start += chunksize


@dispatch(Transpose, np.ndarray)
def compute_up(expr, x, **kwargs):
    return np.transpose(x, axes=expr.axes)


@dispatch(TensorDot, np.ndarray, np.ndarray)
def compute_up(expr, lhs, rhs, **kwargs):
    return np.tensordot(lhs, rhs, axes=[expr._left_axes, expr._right_axes])


@dispatch(IsIn, np.ndarray)
def compute_up(expr, data, **kwargs):
    return np.in1d(data, tuple(expr._keys))


@compute_up.register(Join, DataFrame, np.ndarray)
@compute_up.register(Join, np.ndarray, DataFrame)
@compute_up.register(Join, np.ndarray, np.ndarray)
def join_ndarray(expr, lhs, rhs, **kwargs):
    if isinstance(lhs, np.ndarray):
        lhs = DataFrame(lhs)
    if isinstance(rhs, np.ndarray):
        rhs = DataFrame(rhs)
    return compute_up(expr, lhs, rhs, **kwargs)


@dispatch(Coerce, np.ndarray)
def compute_up(expr, data, **kwargs):
    return data.astype(to_numpy_dtype(expr.schema))


@dispatch(Concat, np.ndarray, np.ndarray)
def compute_up(expr, lhs, rhs, _concat=np.concatenate, **kwargs):
    return _concat((lhs, rhs), axis=expr.axis)


compute_up.register(Coalesce, np.ndarray, (np.ndarray, base))(array_coalesce)
compute_up.register(Coalesce, base, np.ndarray)(array_coalesce)


@dispatch(Coalesce, np.ndarray)
def compute_up(t, data, **kwargs):
    if isinstance(t.lhs, Expr):
        lhs = data
        rhs = t.rhs
    else:
        lhs = t.lhs
        rhs = data

    return array_coalesce(t, lhs, rhs)
