# -*- coding: utf-8 -*-

"""
JIT evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import

import blaze
from blaze.datadescriptor import DyNDDataDescriptor, broadcast_ckernel
from ..pipeline import run_pipeline
from ..passes import ckernel, allocation

from pykit.ir import visit, copy_function
from dynd import nd

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    func, env = run_pipeline(func, env, compile_time_passes)
    return func, env

def interpret(func, env, args, **kwds):
    assert len(args) == len(func.args)

    # Make a copy, since we're going to mutate our IR!
    func = copy_function(func)

    # Update environment with dynd type information
    dynd_types = dict((arg, get_dynd_type(array))
                          for arg, array in zip(func.args, args)
                              if isinstance(array._data, DyNDDataDescriptor))
    env['dynd-types'] = dynd_types

    # Lift ckernels
    func, env = run_pipeline(func, env, run_time_passes)

    # Evaluate
    values = dict(zip(func.args, args))
    interp = CKernelInterp(values)
    visit(interp, func)
    return interp.result

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

compile_time_passes = [
    allocation,
]

run_time_passes = [
    ckernel,
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
        deferred_ckernel = op.args[0]
        args = [self.values[arg] for arg in op.args[1]]

        dst = args[0]
        srcs = args[1:]

        dst_descriptor  = dst._data
        src_descriptors = [src._data for src in srcs]
        #ckernel = unbound_ckernel.bind(dst_descriptor, src_descriptors)

        raise NotImplementedError("Build pointers etc")

        broadcast_ckernel.execute_expr_single(
            dst_descriptor, src_descriptors,
            dst.dshape, [src.dshape for src in srcs],
            deferred_ckernel)

        # Operations are rewritten to already refer to 'dst'
        # We are essentially a 'void' operation
        self.values[op] = None

    def op_ret(self, op):
        retvar = op.args[0]
        self.result = self.values[retvar]

#------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------

def get_dynd_type(array):
    return nd.type_of(array._data.dynd_arr())