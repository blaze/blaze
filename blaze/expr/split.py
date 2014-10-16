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
    chunk = chunk or Symbol('chunk', leaf.dshape)
    if iscollection(center.dshape):
        agg = agg or Symbol('aggregate', center.dshape)
    else:
        agg = agg or Symbol('aggregate', datashape.var * center.dshape)

    ((chunk, chunk_expr), (agg, agg_expr)) = \
            _split(center, leaf=leaf, chunk=chunk, agg=agg, **kwargs)

    return ((chunk, chunk_expr),
            (agg, expr._subs({center: agg})._subs({agg: agg_expr})))


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}


@dispatch(tuple(reductions))
def _split(expr, leaf=None, chunk=None, agg=None, keepdims=True):
    a, b = reductions[type(expr)]
    return ((chunk, a(expr._subs({leaf: chunk})._child, keepdims=keepdims)),
            (agg, b(agg)))


@dispatch(Distinct)
def _split(expr, leaf=None, chunk=None, agg=None):
    return ((chunk, expr._subs({leaf: chunk})),
            (agg, agg.distinct()))


@dispatch(Summary)
def _split(expr, leaf=None, chunk=None, agg=None, keepdims=True):
    chunk_expr = summary(keepdims=keepdims,
                         **dict((name, split(leaf, val, chunk=chunk,
                                             keepdims=False)[0][1])
                            for name, val in zip(expr.fields, expr.values)))
    agg_expr = summary(**dict((name, split(leaf, val, agg=agg,
                                           keepdims=False)[1][1]._subs(
                                                        {agg: agg[name]}))
                            for name, val in zip(expr.fields, expr.values)))
    return ((chunk, chunk_expr), (agg, agg_expr))


@dispatch(By)
def _split(expr, leaf=None, chunk=None, agg=None):
    (chunk, chunk_apply), (agg, agg_apply) = \
            split(leaf, expr.apply, chunk=chunk, agg=agg, keepdims=False)

    chunk_grouper = expr.grouper._subs({leaf: chunk})
    if isscalar(expr.grouper.dshape.measure):
        agg_grouper = agg[expr.fields[0]]
    else:
        agg_grouper = agg[list(expr.fields[:len(expr.grouper.fields)])]

    return ((chunk, by(chunk_grouper, chunk_apply)),
            (agg, by(agg_grouper, agg_apply)))


@dispatch((ElemWise, Selection))
def _split(expr, leaf=None, chunk=None, agg=None):
    return ((chunk, expr._subs({leaf: chunk})),
            (agg, agg))
