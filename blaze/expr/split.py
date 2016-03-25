"""
Expression splitting for chunked computation

To evaluate an expression on a large dataset we may need to chunk that dataset
into pieces and evaluate on each of the pieces individually.  This module
contains logic to break up an expression-to-be-evaluated-on-the-entire-array
into

1.  An expression to be evaluated on each chunk of the data
2.  An expression to be evaluated on the concatenated intermediate results

As an example, consider counting the number of non-null elements in a large
list.  We have the following recipe

1.  Break the large collection into chunks, each of which fits comfortably into
    memory
2.  For each chunk, call compute chunk.count()
3.  Gather all of these results into a single list (which hopefully fits in
    memory)
4.  for this aggregate, compute aggregate.sum()

And so, given this expression

    expr -> expr.count()

We needed the following expressions

    chunk -> chunk.count()
    agg -> agg.sum()

This module performs this transformation for a wide array of chunkable
expressions.  It supports elementwise operations, reductions,
split-apply-combine, and selections.  It notably does not support sorting,
joining, or slicing.

If explicit chunksizes are given it can also reason about the size and shape of
the intermediate aggregate.  It can also do this in N-Dimensions.
"""
from __future__ import absolute_import, division, print_function

from toolz import concat as tconcat
import datashape
from datashape.predicates import isscalar, isrecord
from math import floor

from .core import *
from .expressions import *
from .strings import Like
from .expressions import shape
from .math import sqrt
from .reductions import *
from .split_apply_combine import *
from .collections import *
from ..dispatch import dispatch
from ..compatibility import builtins

good_to_split = (Reduction, Summary, By, Distinct)
can_split = good_to_split + (Like, Selection, ElemWise, Apply)

__all__ = ['path_split', 'split']

def path_split(leaf, expr):
    """ Find the right place in the expression tree/line to parallelize

    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')

    >>> path_split(t, t.amount.sum() + 1)
    sum(t.amount)

    >>> path_split(t, t.amount.distinct().sort())
    distinct(t.amount)
    """
    last = None
    for node in list(path(expr, leaf))[:-1][::-1]:
        if isinstance(node, good_to_split):
            return node
        elif not isinstance(node, can_split):
            return last
        last = node
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

    >>> t = symbol('t', 'var * {name: string, amount: int, id: int}')
    >>> expr = t.id.count()
    >>> split(t, expr)
    ((<`chunk` symbol; dshape='...'>, count(chunk.id, keepdims=True)), (<`aggregate` symbol; dshape='...'>, sum(aggregate)))
    """
    center = path_split(leaf, expr)
    if chunk is None:
        if leaf.ndim > 1:
            raise ValueError("Please provide a chunk symbol")
        else:
            chunk = symbol('chunk', datashape.var * leaf.dshape.measure)

    chunk_expr = _split_chunk(center, leaf=leaf, chunk=chunk, **kwargs)
    chunk_expr_with_keepdims = _split_chunk(center, leaf=leaf, chunk=chunk,
                                            keepdims=True)

    if agg is None:
        agg_shape = aggregate_shape(leaf, expr, chunk, chunk_expr_with_keepdims)
        agg_dshape = DataShape(*(agg_shape + (chunk_expr.dshape.measure,)))
        agg = symbol('aggregate', agg_dshape)

    agg_expr = _split_agg(center, leaf=leaf, agg=agg)

    return ((chunk, chunk_expr),
            (agg, expr._subs({center: agg})._subs({agg: agg_expr})))


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all),
              nelements: (nelements, sum)}

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


@dispatch(mean)
def _split_chunk(expr, leaf=None, chunk=None, keepdims=True):
    child = expr._subs({leaf: chunk})._child
    return summary(total=child.sum(), count=child.count(), keepdims=keepdims,
            axis=expr.axis)

@dispatch(mean)
def _split_agg(expr, leaf=None, agg=None):
    total = agg.total.sum(axis=expr.axis, keepdims=expr.keepdims)
    count = agg['count'].sum(axis=expr.axis, keepdims=expr.keepdims)

    return total / count

@dispatch((std, var))
def _split_chunk(expr, leaf=None, chunk=None, keepdims=True):
    child = expr._subs({leaf: chunk})._child
    return summary(x=child.sum(), x2=(child**2).sum(), n=child.count(),
            keepdims=keepdims, axis=expr.axis)

@dispatch(var)
def _split_agg(expr, leaf=None, agg=None):
    x = agg.x.sum(axis=expr.axis, keepdims=expr.keepdims)
    x2 = agg.x2.sum(axis=expr.axis, keepdims=expr.keepdims)
    n = agg.n.sum(axis=expr.axis, keepdims=expr.keepdims)

    result = (x2 / n) - (x / n)**2

    if expr.unbiased:
        result = result / (n - 1) * n

    return result

@dispatch(std)
def _split_agg(expr, leaf=None, agg=None):
    x = agg.x.sum(axis=expr.axis, keepdims=expr.keepdims)
    x2 = agg.x2.sum(axis=expr.axis, keepdims=expr.keepdims)
    n = agg.n.sum(axis=expr.axis, keepdims=expr.keepdims)

    result = (x2 / n) - (x / n)**2

    if expr.unbiased:
        result = result / (n - 1) * n

    return sqrt(result)

@dispatch(Distinct)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    return expr._subs({leaf: chunk})

@dispatch(Distinct)
def _split_agg(expr, leaf=None, agg=None):
    return agg.distinct()


@dispatch(nunique)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    return (expr._child
                ._subs({leaf: chunk})
                .distinct())

@dispatch(nunique)
def _split_agg(expr, leaf=None, agg=None):
    return agg.distinct().count(keepdims=expr.keepdims)


@dispatch(Summary)
def _split_chunk(expr, leaf=None, chunk=None, keepdims=True):
    exprs = [(name, split(leaf, val, chunk=chunk,
                                       keepdims=False)[0][1])
                            for name, val in zip(expr.fields, expr.values)]
    d = dict()
    for name, e in exprs:
        if isinstance(e, Reduction):
            d[name] = e
        elif isinstance(e, Summary):
            for n, v in zip(e.names, e.values):
                d[name + '_' + n] = v
        else:
            raise NotImplementedError()
    return summary(keepdims=keepdims, **d)


@dispatch(Summary)
def _split_agg(expr, leaf=None, chunk=None, agg=None, keepdims=True):
    exprs = [(name, split(leaf, val, keepdims=False)[1])
                for name, val in zip(expr.fields, expr.values)]

    d = dict()
    for name, (a, ae) in exprs:
        if isscalar(a.dshape.measure): # For simple reductions
            d[name] = ae._subs({a: agg[name]})
        elif isrecord(a.dshape.measure):  # For reductions like mean/var
            names = ['%s_%s' % (name, field) for field in a.fields]
            namedict = dict(zip(a.fields, names))
            d[name] = ae._subs(toolz.merge({a: agg}, namedict))

    return summary(**d)


@dispatch(By)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    chunk_apply = _split_chunk(expr.apply, leaf=leaf, chunk=chunk, keepdims=False)
    chunk_grouper = expr.grouper._subs({leaf: chunk})

    return by(chunk_grouper, chunk_apply)

@dispatch(By)
def _split_agg(expr, leaf=None, agg=None):
    agg_apply = _split_agg(expr.apply, leaf=leaf, agg=agg)
    agg_grouper = expr.grouper._subs({leaf: agg})

    ngroup = len(expr.grouper.fields)

    if isscalar(expr.grouper.dshape.measure):
        agg_grouper = agg[agg.fields[0]]
    else:
        agg_grouper = agg[list(agg.fields[:ngroup])]

    return (by(agg_grouper, agg_apply)
              .relabel(dict(zip(agg.fields[:ngroup],
                                expr.fields[:ngroup]))))


@dispatch((ElemWise, Like, Selection))
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    return expr._subs({leaf: chunk})

@dispatch((ElemWise, Like, Selection))
def _split_agg(expr, leaf=None, agg=None):
    return agg


@dispatch(Apply)
def _split_chunk(expr, leaf=None, chunk=None, **kwargs):
    if expr._splittable:
        return expr._subs({leaf: chunk})
    else:
        raise NotImplementedError()

@dispatch(Apply)
def _split_agg(expr, leaf=None, agg=None):
    return agg


from datashape import Fixed
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
    if a == datashape.var or b == datashape.var:
        return datashape.var
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
    >>> dimension_mul(datashape.var, 5)
    Var()
    >>> dimension_mul(10, datashape.var)
    Var()
    """
    if a == datashape.var or b == datashape.var:
        return datashape.var
    if isinstance(a, Fixed):
        a = int(a)
    if isinstance(b, Fixed):
        b = int(b)
    return int(a * b)


def aggregate_shape(leaf, expr, chunk, chunk_expr):
    """ The shape of the intermediate aggregate

    >>> leaf = symbol('leaf', '10 * 10 * int')
    >>> expr = leaf.sum(axis=0)
    >>> chunk = symbol('chunk', '3 * 3 * int') # 3 does not divide 10
    >>> chunk_expr = chunk.sum(axis=0, keepdims=True)

    >>> aggregate_shape(leaf, expr, chunk, chunk_expr)
    (4, 10)
    """
    if datashape.var in tconcat(map(shape, [leaf, expr, chunk, chunk_expr])):
        return (datashape.var, ) * leaf.ndim

    numblocks = [int(floor(l / c)) for l, c in zip(leaf.shape, chunk.shape)]
    last_chunk_shape = [l % c for l, c in zip(leaf.shape, chunk.shape)]

    if builtins.sum(last_chunk_shape) != 0:
        old = last_chunk_shape
        last_chunk = symbol(chunk._name,
                            DataShape(*(last_chunk_shape + [chunk.dshape.measure])))
        last_chunk_expr = chunk_expr._subs({chunk: last_chunk})
        last_chunk_shape = shape(last_chunk_expr)
        # Keep zeros if they were there before
        last_chunk_shape = tuple(a if b != 0 else 0 for a, b in
                                zip(last_chunk_shape, old))


    return tuple(int(floor(l / c)) * ce + lce
            for l, c, ce, lce
            in zip(shape(leaf), shape(chunk), shape(chunk_expr), last_chunk_shape))
