from __future__ import absolute_import, division, print_function

from .core import *
from .expressions import *
from .reductions import *
from .split_apply_combine import *
from .collections import *
from .table import *
import datashape
from datashape.predicates import isscalar
from ..dispatch import dispatch

good_to_split = (Reduction, Summary, By, Distinct)
can_split = good_to_split + (Selection, ElemWise)

__all__ = ['path_split', 'split']

def path_split(leaf, expr):
    """ Find the right place in the expression tree/line to parallelize

    >>> t = Symbol('t', 'var * {name: string, amount: int, id: int}')

    >>> path_split(t, t.amount.sum() + 1)
    sum(_child=t.amount, axis=(0,), keepdims=False)

    >>> path_split(t, t.amount.distinct().sort())
    Distinct(_child=t.amount)
    """
    last = None
    for node in list(path(expr, leaf))[:-1][::-1]:
        if isinstance(node, good_to_split):
            return node
        elif not isinstance(node, can_split):
            return last
        last = node
    if not node:
        raise ValueError()
    return node


def split(leaf, expr, chunk=None, agg=None, **kwargs):
    """ Split expression for chunked computation

    Break up a computation ``leaf -> expr`` so that it can be run in chunks.
    This returns two computations, one to perform on each chunk and then one to
    perform on the union of these intermediate results

        chunk -> chunk-expr
        aggregate -> aggregate-expr

    The chunk_expr will have the same dimesions as the input
    (reductions set keepdims=True) so that this should work cleanly with
    concatenation functions like ``np.concatenate``.

    Returns
    -------

    Pair of (Symbol, Expr) pairs

        (chunk, chunk_expr), (aggregate, aggregate_expr)

    >>> t = Symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> expr = t.id.count()
    >>> split(t, expr)
    ((chunk, count(_child=chunk.id, axis=(0,), keepdims=True)), (aggregate, sum(_child=aggregate, axis=(0,), keepdims=False)))
    """
    center = path_split(leaf, expr)
    if not chunk:
        if leaf.ndim > 1:
            raise ValueError("Please provide a chunk symbol")
        else:
            chunk = Symbol('chunk', datashape.var * leaf.dshape.measure)

    chunk_expr = _split_chunk(center, leaf=leaf, chunk=chunk, **kwargs)

    if not agg:
        blocks_shape = tuple(map(dimension_div, leaf.shape, chunk.shape))
        agg_shape = tuple(map(dimension_mul, blocks_shape, shape(chunk_expr)))
        agg_dshape = DataShape(*(agg_shape + (chunk_expr.dshape.measure,)))

        agg = Symbol('aggregate', agg_dshape)

    agg_expr = _split_agg(center, leaf=leaf, agg=agg)

    return ((chunk, chunk_expr),
            (agg, expr._subs({center: agg})._subs({agg: agg_expr})))


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}

@dispatch(Expr)
def _split(expr, leaf=None, chunk=None, agg=None, keepdims=True):
    return ((chunk, _split_chunk(expr, leaf=leaf, chunk=chunk,
                                 keepdims=keepdims)),
            (agg, _split_agg(expr, leaf=leaf, agg=agg)))


@dispatch(tuple(reductions))
def _split_chunk(expr, leaf=None, chunk=None, keepdims=True):
    a, b = reductions[type(expr)]
    return a(expr._subs({leaf: chunk})._child,
             keepdims=keepdims,
             axis=expr.axis)

@dispatch(tuple(reductions))
def _split_agg(expr, leaf=None, agg=None):
    a, b = reductions[type(expr)]
    return b(agg, axis=expr.axis, keepdims=expr.keepdims)


@dispatch(Distinct)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    return expr._subs({leaf: chunk})

@dispatch(Distinct)
def _split_agg(expr, leaf=None, agg=None):
    return agg.distinct()


@dispatch(Summary)
def _split_chunk(expr, leaf=None, chunk=None, keepdims=True):
    return summary(keepdims=keepdims,
                   **dict((name, split(leaf, val, chunk=chunk,
                                       keepdims=False)[0][1])
                            for name, val in zip(expr.fields, expr.values)))
    return chunk_expr


@dispatch(Summary)
def _split_agg(expr, leaf=None, chunk=None, agg=None, keepdims=True):
    return summary(**dict((name, split(leaf, val, agg=agg,
                                       keepdims=False)[1][1]._subs(
                                                        {agg: agg[name]}))
                            for name, val in zip(expr.fields, expr.values)))


@dispatch(By)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    chunk_apply = _split_chunk(expr.apply, leaf=leaf, chunk=chunk, keepdims=False)
    chunk_grouper = expr.grouper._subs({leaf: chunk})

    return by(chunk_grouper, chunk_apply)

@dispatch(By)
def _split_agg(expr, leaf=None, agg=None):
    agg_apply = _split_agg(expr.apply, leaf=leaf, agg=agg)
    agg_grouper = expr.grouper._subs({leaf: agg})

    if isscalar(expr.grouper.dshape.measure):
        agg_grouper = agg[expr.fields[0]]
    else:
        agg_grouper = agg[list(expr.fields[:len(expr.grouper.fields)])]

    return by(agg_grouper, agg_apply)


@dispatch((ElemWise, Selection))
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    return expr._subs({leaf: chunk})

@dispatch((ElemWise, Selection))
def _split_agg(expr, leaf=None, agg=None):
    return agg


from datashape import var, Fixed
from math import ceil

def dimension_div(a, b):
    """ How many times does b fit into a?

    >>> dimension_div(10, 5)
    2

    We round up
    >>> dimension_div(20, 9)
    3

    In the case of datashape.var, we resort to var
    >>> from datashape import var
    >>> dimension_div(var, 5)
    Var()
    >>> dimension_div(50, var)
    Var()
    """
    if a == var or b == var:
        return var
    if isinstance(a, Fixed):
        a = int(a)
    if isinstance(b, Fixed):
        b = int(b)
    return int(ceil(a / b))


def dimension_mul(a, b):
    """ Given b number of a's how big is our dimension?

    >>> dimension_mul(2, 5)
    10

    We round up
    >>> dimension_mul(9, 3)
    27

    In the case of datashape.var, we resort to var
    >>> from datashape import var
    >>> dimension_mul(var, 5)
    Var()
    >>> dimension_mul(10, var)
    Var()
    """
    if a == var or b == var:
        return var
    if isinstance(a, Fixed):
        a = int(a)
    if isinstance(b, Fixed):
        b = int(b)
    return int(a * b)
