"""
Graph expression context.
"""

import blaze
from blaze.datashape import unify

class ExprContext(object):
    """Context for blaze graph expressions"""

    def __init__(self):
        # Coercion constraints between types with free variables
        self.constraints = []
        self.inputs = []      # Accumulated input parameters (arrays)


def merge(contexts):
    """
    Merge graph expression contexts into a new context, unifying their
    typing contexts under the given blaze function signature.
    """
    result = ExprContext()

    for ctx in contexts:
        result.constraints.extend(ctx.constraints)
        result.inputs.extend(ctx.inputs)

    result.constraints, _ = unify(result.constraints)
    return result

def initialize(context, term):
    """Initialize an expression context with a graph term"""
    if isinstance(term, blaze.Array):
        context.inputs.append(term)
