from __future__ import absolute_import, division, print_function

from blaze.expr import *
from blaze.expr.table import *
import datashape
from datashape.predicates import isscalar
from ..dispatch import dispatch

good_to_split = (Reduction, Summary, By, Distinct)
can_split = good_to_split + (Selection, ElemWise)

def path_split(leaf, expr):
    """ Find the right place in the expression tree/line to parallelize

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')

    >>> path_split(t, t.amount.sum() + 1)
    sum(child=t['amount'])

    >>> path_split(t, t.amount.distinct().sort())
    Distinct(child=t['amount'])
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


def split(leaf, expr, chunk=None, agg=None):
    """ Split expression for chunked computation

    Break up a computation ``leaf -> expr`` so that it can be run in chunks.
    This returns two computations, one to perform on each chunk and then one to
    perform on the union of these intermediate results

        chunk -> chunk-expr
        aggregate -> aggregate-expr

    Returns
    -------

    Pair of (TableSymbol, Expr) pairs

        (chunk, chunk_expr), (aggregate, aggregate_expr)

    >>> t = TableSymbol('t', '{name: string, amount: int, id: int}')
    >>> expr = t.id.count()
    >>> split(t, expr)
    ((chunk, count(child=chunk['id'])), (aggregate, sum(child=aggregate)))
    """
    center = path_split(leaf, expr)
    chunk = chunk or TableSymbol('chunk', leaf.dshape)
    agg = agg or TableSymbol('aggregate', center.dshape)

    ((chunk, chunk_expr), (agg, agg_expr)) = \
            _split(center, leaf=leaf, chunk=chunk, agg=agg)

    return ((chunk, chunk_expr),
            (agg, expr.subs({center: agg}).subs({agg: agg_expr})))


reductions = {sum: (sum, sum), count: (count, sum),
              min: (min, min), max: (max, max),
              any: (any, any), all: (all, all)}


@dispatch(tuple(reductions))
def _split(expr, leaf=None, chunk=None, agg=None):
    a, b = reductions[type(expr)]
    return ((chunk, a(expr.subs({leaf: chunk}).child)),
            (agg, b(agg)))


@dispatch(Distinct)
def _split(expr, leaf=None, chunk=None, agg=None):
    return ((chunk, expr.subs({leaf: chunk})),
            (agg, agg.distinct()))


@dispatch(Summary)
def _split(expr, leaf=None, chunk=None, agg=None):
    chunk_expr = summary(**dict((name, split(leaf, val, chunk=chunk)[0][1])
                            for name, val in zip(expr.fields, expr.values)))
    agg_expr = summary(**dict((name, split(leaf, val, agg=agg)[1][1].subs(
                                                        {agg: agg[name]}))
                            for name, val in zip(expr.fields, expr.values)))
    return ((chunk, chunk_expr), (agg, agg_expr))


@dispatch(By)
def _split(expr, leaf=None, chunk=None, agg=None):
    (chunk, chunk_apply), (agg, agg_apply) = \
            split(leaf, expr.apply, chunk=chunk, agg=agg)

    chunk_grouper = expr.grouper.subs({leaf: chunk})
    if isscalar(expr.grouper.dshape.measure):
        agg_grouper = agg[expr.columns[0]]
    else:
        agg_grouper = agg[list(expr.columns[:len(expr.grouper.columns)])]

    return ((chunk, by(chunk_grouper, chunk_apply)),
            (agg, by(agg_grouper, agg_apply)))


@dispatch((ElemWise, Selection))
def _split(expr, leaf=None, chunk=None, agg=None):
    return ((chunk, expr.subs({leaf: chunk})),
            (agg, agg))
