from blaze.expr import *
import datashape

good_to_split = (Reduction, By, Distinct)
can_split = good_to_split + (Selection, RowWise)

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
    return node


def split(leaf, expr):
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
    chunk = TableSymbol('chunk', leaf.dshape, leaf.iscolumn)
    if isinstance(center, TableExpr):
        agg = TableSymbol('aggregate', center.schema, center.iscolumn)
    else:
        agg = TableSymbol('aggregate', datashape.var * center.dshape, True)

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
