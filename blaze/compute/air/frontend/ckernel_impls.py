"""
Convert 'kernel' Op to 'ckernel'.
"""

from __future__ import absolute_import, division, print_function

from pykit.ir import transform, Op


def run(func, env):
    strategies = env['strategies']
    transform(CKernelImplementations(strategies), func)


class CKernelImplementations(object):
    """
    For kernels that are implemented via ckernels, this
    grabs the ckernel_deferred and turns it into a ckernel
    op.
    """

    def __init__(self, strategies):
        self.strategies = strategies

    def op_kernel(self, op):
        if self.strategies[op] != 'ckernel':
            return

        # Default overload is CKERNEL, so no need to look it up again
        overload = op.metadata['overload']

        impl = overload.func

        new_op = Op('ckernel', op.type, [impl, op.args[1:]], op.result)
        new_op.add_metadata({'rank': 0,
                             'parallel': True})
        return new_op
