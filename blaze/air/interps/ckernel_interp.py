# -*- coding: utf-8 -*-

"""
JIT evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import

import blaze
from ..pipeline import run_pipeline
from ..passes import ckernel, allocation

from pykit.ir import visit

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    func, env = run_pipeline(func, env, passes)
    return func, env

def interpret(func, args, **kwds):
    assert len(args) == len(func.args)
    values = dict(zip(func.args, args))
    visit(CKernelInterp(values), func)

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

passes = [
    ckernel,
    allocation,
]

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

class CKernelInterp(object):
    """
    Interpret low-level AIR in the most straightforward way possible.

    Low-level AIR contains the following operations:

        alloc/dealloc
        ckernel

    There is a huge number of things we can still do, like blocking and
    parallelism.

    Blocking
    ========
    This should probably happen through a "blocking-ckernel" wrapper

    Parallelism
    ===========
    Both data-parallelism by executing ckernels over slices, and executing
    disjoint sub-expressions in parallel.
    """

    def __init__(self, values):
        self.values = values # { Op : py_val }

    def op_alloc(self, op):
        dshape = op.type
        storage = op.metadata.get('storage') # TODO: storage!
        self.values[op] = blaze.empty(dshape, storage=storage)

    def op_dealloc(self, op):
        alloc, last_op = op.args
        del self.values[alloc]

    def op_ckernel(self, op):
        ckernel = op.args[0]
        args = [self.values[arg] for arg in op.args[1]]
        self.values[op] = ckernel(*args)

