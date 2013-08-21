"""
Graph expression context.
"""

from blaze import array
from blaze.datashape import coretypes

class ExprContext(object):
    """Context for blaze graph expressions"""

    def __init__(self):
        self.constraints = {} # Type equations
        self.solution = {}    # Partial typing solution, TypeVar -> Type
        self.inputs = []      # Accumulated input parameters (arrays)

def unify(signature, *contexts):
    """
    Merge graph expression contexts into a new context, unifying their
    typing contexts under the given blaze function signature.
    """
    ## TODO: finish

    result = ExprContext()

    for ctx in contexts:
        solution = unify(result.constraints, ctx.constraints)
        result.constraints.update(solution)
        result.inputs.extend(ctx.inputs)

    return result

def initialize(context, term):
    """Initialize an expression context with a graph term"""
    if isinstance(term, array.Array):
        context.inputs.append(term)