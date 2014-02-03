"""Graph expression context."""

from __future__ import absolute_import, division, print_function

from datashape import unify


class ExprContext(object):
    """
    Context for blaze graph expressions.

    This keeps track of a mapping between graph expression nodes and the
    concrete data inputs (i.e. blaze Arrays).

    Attributes:
    ===========

    terms: { ArrayOp: Array }
        Mapping from ArrayOp nodes to inputs
    """

    def __init__(self):
        # Coercion constraints between types with free variables
        self.constraints = []
        self.terms = {} # All terms in the graph, { Array : Op }
        self.params = []

    def add_input(self, term, data):
        if term not in self.terms:
            self.params.append(term)
        self.terms[term] = data


def merge(contexts):
    """
    Merge graph expression contexts into a new context, unifying their
    typing contexts under the given blaze function signature.
    """
    result = ExprContext()

    for ctx in contexts:
        result.constraints.extend(ctx.constraints)
        result.terms.update(ctx.terms)
        result.params.extend(ctx.params)

    result.constraints, _ = unify(result.constraints)
    return result
