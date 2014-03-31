"""
Rewrite ckernels to executable pykernels.
"""

from __future__ import absolute_import, division, print_function

from dynd import nd, ndt

from ..ir import Op


def run(func, env):
    storage = env['storage']

    for op in func.ops:
        if op.opcode == 'ckernel':
            # Build an executable chunked or in-memory pykernel
            if storage is None:
                pykernel = op_ckernel(op)
            else:
                pykernel = op_ckernel_chunked(op)

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

        dst_descriptor  = dst.ddesc
        src_descriptors = [src.ddesc for src in srcs]

        out = dst_descriptor.dynd_arr()
        inputs = [desc.dynd_arr() for desc in src_descriptors]

        # Execute!
        deferred_ckernel.__call__(out, *inputs)

    return pykernel


def op_ckernel_chunked(op):
    """
    Create a pykernel for a ckernel for uniform interpretation that handled
    chunked out-of-core execution.
    """
    deferred_ckernel = op.args[0]

    def pykernel(*args):
        dst = args[0]
        srcs = args[1:]

        dst_descriptor  = dst.ddesc
        src_descriptors = [src.ddesc for src in srcs]

        out = dst_descriptor.dynd_arr()
        inputs = [desc.dynd_arr() for desc in src_descriptors]

        # TODO: Remove later, explicit casting necessary for now because
        #       of BLZ/numpy interop effect.
        for i, (inp, tp) in enumerate(zip(inputs, deferred_ckernel.types[1:])):
            tp = ndt.type(tp)
            if nd.type_of(inp) != tp:
                inputs[i] = nd.array(inp, type=tp)

        # Execute!
        deferred_ckernel.__call__(out, *inputs)

    return pykernel
