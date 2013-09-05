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
        self.inputs = {}      # Accumulated input parameters (arrays)
        self.params = []

    def add_input(self, term, data):
        if term not in self.inputs:
            self.params.append(term)
        self.inputs[term] = data


def merge(contexts):
    """
    Merge graph expression contexts into a new context, unifying their
    typing contexts under the given blaze function signature.
    """
    result = ExprContext()

    for ctx in contexts:
        result.constraints.extend(ctx.constraints)
        result.inputs.update(ctx.inputs)
        result.params.extend(ctx.params)

    result.constraints, _ = unify(result.constraints)
    return result
