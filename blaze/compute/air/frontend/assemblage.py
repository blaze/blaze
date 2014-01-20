# -*- coding: utf-8 -*-

"""
Assemble kernels into.
"""

from __future__ import absolute_import, division, print_function
from pykit.ir import Op

def assemble_py_kernels(func, env):
    for op in func.ops:
        if op.opcode == 'kernel':
            kernel   = op.metadata['kernel']
            overload = op.metadata['overload']

            func = overload.func

            args = op.args[1:]
            py_kernel = Op('pykernel', op.type, [func, args], op.result)
            op.replace(py_kernel)
