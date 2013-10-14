# -*- coding: utf-8 -*-

"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import transform, Op
from dynd import nd, ndt, _lowlevel

#------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------

def run(func, env):
    transform(CKernelImplementations(), func)

#------------------------------------------------------------------------
# Extract CKernel Implementations
#------------------------------------------------------------------------

class CKernelImplementations(object):
    """
    For kernels that are implemented via ckernels, this
    grabs the ckernel_deferred and turns it into a ckernel
    op.
    """
    def op_kernel(self, op):
        function = op.metadata['kernel']
        overload = op.metadata['overload']
        func = overload.func
        polysig = overload.sig
        monosig = overload.resolved_sig
        impls = function.find_impls(func, polysig, 'ckernel')
        if impls:
            [impl] = impls
            new_op = Op('ckernel', op.type, [impl, op.args[1:]], op.result)
            new_op.add_metadata({'rank': 0,
                                 'parallel': True})
            return new_op
        return op
