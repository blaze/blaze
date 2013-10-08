# -*- coding: utf-8 -*-

"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import visit
from dynd import nd, ndt, _lowlevel

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
        if op.metadata['rank'] < len(op.type.shape):
            ckernel, args = op.args
            in_dshapes = [arg.type for arg in args[1:]]
            out_dshape = args[0].type
            new_ckernel = lift_ckernel(ckernel, out_dshape, in_dshapes)
            op.args[0] = new_ckernel


def lift_ckernel(ckernel, out_dshape, in_dshapes):
    lifted_types = [ndt.type(str(ds)) for ds in [out_dshape] + in_dshapes]
    print("original types:", repr(ckernel.types))
    print("lifted types:", lifted_types)
    return _lowlevel.lift_ckernel_deferred(ckernel, lifted_types)
