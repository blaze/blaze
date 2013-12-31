"""
Blaze AIR annotators that add some information to the IR.
"""

from __future__ import absolute_import, division, print_function

#------------------------------------------------------------------------
# Annotators
#------------------------------------------------------------------------

def annotate_uses(func, values):
    """
    Annotate the Blaze AIR function with metadata indicating the
    number of external uses. These points need to have concrete arrays
    associated with it and represent fusion boundaries.

    Parameters
    ----------
    func: pykit.ir.Function
        Typed Blaze function

    values: { pykit.ir.Operation : blaze.expr.Op }
        Value map for kernel Ops
    """
    exprs = set(values.values())
    for op in func.ops:
        if op.opcode == 'kernel':
            expr = values[op]
            internal = sum(use in exprs for use in expr.uses)
            external = len(expr.uses) - internal
            op.add_metadata(external_uses=external)