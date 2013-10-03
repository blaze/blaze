# -*- coding: utf-8 -*-

"""
Insert temporary allocations and deallocations into the IR.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import interp, visit, transform, Op, Builder
from pykit import types

def run(func, env):
    b = Builder(func)

    # IR positions and list of ops
    positions = dict((op, idx) for idx, op in enumerate(func.ops))
    ops = list(func.ops)

    for op in func.ops:
        if op.opcode == 'ckernel':
            alloc   = Op('alloc', op.type, args=[])

            # TODO: Insert alloc in args list of ckernel

            # Replace uses of ckernel with temporary allocation
            op.replace_uses(alloc)

            # Emit allocation before first use
            b.position_before(op)
            b.emit(alloc)

            # Emit deallocation after last use
            idx = max(positions[u] for u in func.uses[op])
            last_op = ops[idx]
            b.position_after(last_op)
            dealloc = Op('dealloc', types.Void, [alloc, last_op])
            b.emit(dealloc)

    return func
