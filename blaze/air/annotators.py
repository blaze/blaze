# -*- coding: utf-8 -*-

"""
Blaze AIR annotators that add some information to the IR.
"""

from __future__ import print_function, division, absolute_import

#------------------------------------------------------------------------
# Annotators
#------------------------------------------------------------------------

def annotate_uses(func, values):
    """
    Annotate the Blaze AIR function with metadata indicating the
    number of external uses. These points need to have data associated with
    it.

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