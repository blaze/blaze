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
    visit(CKernelLifter(env), func)

#------------------------------------------------------------------------
# Lift CKernels
#------------------------------------------------------------------------

class CKernelLifter(object):
    """
    Lift ckernels to their appropriate rank so they always consume the
    full array arguments.
    """
    def __init__(self, env):
        self.env = env

    def get_arg_type(self, arg):
        dynd_types = self.env['dynd-types']
        if arg in dynd_types:
            return dynd_types[arg]
        else:
            return ndt.type(str(arg.type))

    def op_ckernel(self, op):
        if op.metadata['rank'] < len(op.type.shape):
            ckernel, args = op.args
            in_types = [self.get_arg_type(arg) for arg in args[1:]]
            out_type = ndt.type(str(args[0].type))
            op.args[0] = _lowlevel.lift_ckernel_deferred(ckernel,
                            [out_type] + in_types)
