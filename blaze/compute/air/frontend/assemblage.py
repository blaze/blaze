# -*- coding: utf-8 -*-

"""
Assemble kernels into pykernels for execution.
"""

from __future__ import absolute_import, division, print_function
from pykit.ir import Op
from blaze.compute.strategy import PY

def assemble_kernels(func, env, pykernels, strategy):
    """
    Transforms kernel ops to pykernel ops, for execution by a simple
    interpreter.

    Arguments
    =========
    pykernels: { Op: py_func }
        python applyable kernels that accept blaze arrays

    strategy: str
        the strategy for which we are applying the transform
    """
    strategies = env['strategies']

    for op in func.ops:
        if op.opcode == 'kernel' and strategies[op] == strategy:
            pykernel = pykernels[op]
            op.replace(Op('pykernel', op.type, [pykernel, op.args[1:]],
                          op.result))


def assemble_py_kernels(func, env):
    """kernel('add', a, b) -> pykernel(add, a, b)"""
    overloads = env['kernel.overloads']

    pykernels = dict((op, overloads[op, PY])
                         for op in func.ops if (op, PY) in overloads)
    assemble_kernels(func, env, pykernels, PY)