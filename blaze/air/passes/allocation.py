# -*- coding: utf-8 -*-

"""
Insert temporary allocations and deallocations into the IR.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp, visit, transform, Op, Builder, ops
from pykit import types

def insert_allocations(func, env):
    b = Builder(func)

    # IR positions and list of ops
    positions = dict((op, idx) for idx, op in enumerate(func.ops))
    oplist = list(func.ops)

    for op in func.ops:
        if op.opcode == 'ckernel':
            ckernel, args = op.args
            alloc   = Op('alloc', op.type, args=[])

            # TODO: Insert alloc in args list of ckernel

            # Replace uses of ckernel with temporary allocation
            op.replace_uses(alloc)
            op.set_args([ckernel, [alloc] + args])

            # Emit allocation before first use
            b.position_before(op)
            b.emit(alloc)

            # Emit deallocation after last use, unless we are returning
            # the result
            idx = max(positions[u] for u in func.uses[alloc])
            last_op = oplist[idx]
            if not last_op.opcode == 'ret':
                b.position_after(last_op)
                dealloc = Op('dealloc', types.Void, [alloc, last_op])
                b.emit(dealloc)

    return func, env


run = insert_allocations