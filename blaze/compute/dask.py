from __future__ import absolute_import, division, print_function

from numbers import Number
from toolz import concat, first, curry, compose
from datashape import DataShape

from blaze import compute, ndim
from blaze.dispatch import dispatch
from blaze.compute.core import compute_up, optimize
from blaze.expr import (ElemWise, symbol, Reduction, Transpose, TensorDot,
                        Expr, Slice, Broadcast)
from blaze.expr.split import split

from dask.array.core import _concatenate2, Array, atop, transpose, tensordot


def compute_it(expr, leaves, *data, **kwargs):
    kwargs.pop('scope')
    return compute(expr, dict(zip(leaves, data)), **kwargs)


def elemwise_array(expr, *data, **kwargs):
    leaves = expr._inputs
    expr_inds = tuple(range(ndim(expr)))[::-1]
    return atop(curry(compute_it, expr, leaves, **kwargs),
                expr_inds,
                *concat((dat, tuple(range(ndim(dat))[::-1])) for dat in data))


try:
    from blaze.compute.numba import (get_numba_ufunc, broadcast_collect,
            Broadcastable)

    def compute_broadcast(expr, *data, **kwargs):
        expr_inds = tuple(range(ndim(expr)))[::-1]
        func = get_numba_ufunc(expr)
        return atop(func,
                    expr_inds,
                    *concat((dat, tuple(range(ndim(dat))[::-1])) for dat in data))

    def optimize_array(expr, *data):
        return broadcast_collect(
            expr,
            broadcastable=Broadcastable,
            want_to_broadcast=Broadcastable,
        )

    for i in range(5):
        compute_up.register(Broadcast, *([(Array, Number)] * i))(compute_broadcast)
        optimize.register(Expr, *([(Array, Number)] * i))(optimize_array)

except ImportError:
    pass


for i in range(5):
    compute_up.register(ElemWise, *([Array] * i))(elemwise_array)


@dispatch(Reduction, Array)
def compute_up(expr, data, **kwargs):
    leaf = expr._leaves()[0]
    chunk = symbol('chunk', DataShape(*(tuple(map(first, data.chunks)) +
                                        (leaf.dshape.measure,))))
    (chunk, chunk_expr), (agg, agg_expr) = split(expr._child, expr,
                                                 chunk=chunk)

    inds = tuple(range(ndim(leaf)))
    tmp = atop(curry(compute_it, chunk_expr, [chunk], **kwargs), inds, data,
               inds)

    return atop(compose(curry(compute_it, agg_expr, [agg], **kwargs),
                        curry(_concatenate2, axes=expr.axis)),
                tuple(i for i in inds if i not in expr.axis),
                tmp, inds)


@dispatch(Transpose, Array)
def compute_up(expr, data, **kwargs):
    return transpose(data, expr.axes)


@dispatch(TensorDot, Array, Array)
def compute_up(expr, lhs, rhs, **kwargs):
    return tensordot(lhs, rhs, (expr._left_axes, expr._right_axes))


@dispatch(Slice, Array)
def compute_up(expr, data, **kwargs):
    return data[expr.index]
