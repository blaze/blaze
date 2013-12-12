# -*- coding: utf-8 -*-

"""
Lift ckernels to their appropriate rank so they always consume the full array
arguments.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import transform, Op

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
        argtypes = monosig.argtypes

        if function.matches('ckernel', argtypes):
            overload = function.best_match('ckernel', argtypes)
            impl = overload.func
            assert monosig == overload.resolved_sig, (monosig,
                                                      overload.resolved_sig)

            new_op = Op('ckernel', op.type, [impl, op.args[1:]], op.result)
            new_op.add_metadata({'rank': 0,
                                 'parallel': True})
            return new_op
        return op
