# -*- coding: utf-8 -*-

"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import visit

#------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------

def run(func, env):
    visit(CKernelLifter(), func)

#------------------------------------------------------------------------
# Lift CKernels
#------------------------------------------------------------------------

class CKernelLifter(object):
    """
    Lift ckernels to their appropriate rank so they always consume the
    full array arguments.
    """

    def op_ckernel(self, op):
        if op.metadata['rank'] < op.type.rank:
            ckernel, args = op.args
            in_dshapes = [arg.type for arg in args]
            out_dshape = op.type
            new_ckernel = lift_ckernel(ckernel, out_dshape, in_dshapes)
            op.args[0] = new_ckernel


def lift_ckernel(ckernel, out_dshape, in_dshapes):
    raise NotImplementedError