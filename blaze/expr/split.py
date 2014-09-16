from blaze.expr import *

good_to_split = (Reduction, By)
can_split = good_to_split + (Distinct, Selection, RowWise)

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
