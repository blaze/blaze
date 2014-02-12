# -*- coding: utf-8 -*-

"""
Rewrite ckernels to executable pykernels.
"""

from __future__ import absolute_import, division, print_function

from pykit.ir import Op

def run(func, env):

    for op in func.ops:
        if op.opcode == 'ckernel':
            pykernel = op_ckernel(op)
            newop = Op('pykernel', op.type, [pykernel, op.args[1]],
                       op.result)
            op.replace(newop)


def op_ckernel(op):
    """
    Create a pykernel for a ckernel for uniform interpretation.
    """
    deferred_ckernel = op.args[0]

    def pykernel(*args):
        dst = args[0]
        srcs = args[1:]

        dst_descriptor  = dst._data
        src_descriptors = [src._data for src in srcs]

        out = dst_descriptor.dynd_arr()
        inputs = [desc.dynd_arr() for desc in src_descriptors]

        # Execute!
        deferred_ckernel.__call__(out, *inputs)

    return pykernel