# -*- coding: utf-8 -*-

"""
JIT evaluation of blaze AIR.
"""

from __future__ import print_function, division, absolute_import

import operator
import blaze
from blaze.datadescriptor import DyNDDataDescriptor, BLZDataDescriptor
from ..pipeline import run_pipeline
from ..passes import ckernel_impls, ckernel_lift, allocation

from pykit.ir import visit, copy_function
from dynd import nd, ndt
from blaze import blz

#------------------------------------------------------------------------
# Interpreter
#------------------------------------------------------------------------

def compile(func, env):
    func, env = run_pipeline(func, env, compile_time_passes)

    return func, env

def interpret(func, env, args, storage=None, **kwds):
    assert len(args) == len(func.args)

    # Make a copy, since we're going to mutate our IR!
    func = copy_function(func)

    # If it's a BLZ output, we want an interpreter that streams
    # the processing through in chunks
    if storage is not None:
        if len(func.type.restype.shape) == 0:
            raise TypeError('Require an array, not a scalar, for outputting to BLZ')
        env['stream-outer'] = True
        result_ndim = env['result-ndim'] = len(func.type.restype.shape)
    else:
        # Convert any persistent inputs to memory
        # TODO: should stream the computation in this case
        for i, arg in enumerate(args):
            if isinstance(arg._data, BLZDataDescriptor):
                args[i] = arg[:]

    # Update environment with dynd type information
    dynd_types = dict((arg, get_dynd_type(array))
                          for arg, array in zip(func.args, args)
                              if isinstance(array._data, DyNDDataDescriptor))
    env['dynd-types'] = dynd_types

    # Lift ckernels
    func, env = run_pipeline(func, env, run_time_passes)

    if storage is None:
        # Evaluate once
        values = dict(zip(func.args, args))
        interp = CKernelInterp(values)
        visit(interp, func)
        return interp.result
    else:
        res_shape, res_dt = blaze.datashape.to_numpy(func.type.restype)
        dim_size = operator.index(res_shape[0])
        row_size = ndt.type(str(func.type.restype.subarray(1))).data_size
        chunk_size = min(max(1, (1024*1024) // row_size), dim_size)
        # Evaluate by streaming the outermost dimension,
        # and using the BLZ data descriptor's append
        dst_dd = BLZDataDescriptor(blz.zeros((0,)+res_shape[1:], res_dt,
                                             rootdir=storage.path))
        # Loop through all the chunks
        for chunk_start in range(0, dim_size, chunk_size):
            # Tell the interpreter which chunk size to use (last
            # chunk might be smaller)
            chunk_size = min(chunk_size, dim_size - chunk_start)
            # Evaluate the chunk
            args_chunk = [arg[chunk_start:chunk_start+chunk_size]
                            if len(arg.dshape.shape) == result_ndim
                            else arg for arg in args]
            values = dict(zip(func.args, args_chunk))
            interp = CKernelChunkInterp(values, chunk_size, result_ndim)
            visit(interp, func)
            chunk = interp.result._data.dynd_arr()
            dst_dd.append(chunk)
        return blaze.Array(dst_dd)

#------------------------------------------------------------------------
# Passes
#------------------------------------------------------------------------

compile_time_passes = [
]

run_time_passes = [
    ckernel_impls,
    allocation,
    ckernel_lift,
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
        alloc, = op.args
        del self.values[alloc]

    def op_convert(self, op):
        input = self.values[op.args[0]]
        input = input._data.dynd_arr()
        result = nd.array(input, type=ndt.type(str(op.type)))
        result = blaze.Array(DyNDDataDescriptor(result))
        self.values[op] = result

    def op_ckernel(self, op):
        deferred_ckernel = op.args[0]
        args = [self.values[arg] for arg in op.args[1]]

        dst = args[0]
        srcs = args[1:]

        dst_descriptor  = dst._data
        src_descriptors = [src._data for src in srcs]

        out = dst_descriptor.dynd_arr()
        inputs = [desc.dynd_arr() for desc in src_descriptors]

        # Execute!
        deferred_ckernel.__call__(out, *inputs)

        # Operations are rewritten to already refer to 'dst'
        # We are essentially a 'void' operation
        self.values[op] = None

    def op_ret(self, op):
        retvar = op.args[0]
        self.result = self.values[retvar]


class CKernelChunkInterp(object):
    """
    Like CKernelInterp, but for processing one chunk.
    """

    def __init__(self, values, chunk_size, result_ndim):
        self.values = values # { Op : py_val }
        self.chunk_size = chunk_size
        self.result_ndim = result_ndim

    def op_alloc(self, op):
        dshape = op.type
        # Allocate a chunk instead of the whole thing
        if len(dshape.shape) == self.result_ndim:
            chunk = nd.empty(self.chunk_size, str(dshape.subarray(1)))
        else:
            chunk = nd.empty(str(dshape))
        self.values[op] = blaze.array(chunk)

    def op_dealloc(self, op):
        alloc, = op.args
        del self.values[alloc]

    def op_convert(self, op):
        input = self.values[op.args[0]]
        input = input._data.dynd_arr()
        result = nd.array(input, type=ndt.type(str(op.type)))
        result = blaze.Array(DyNDDataDescriptor(result))
        self.values[op] = result

    def op_ckernel(self, op):
        deferred_ckernel = op.args[0]
        args = [self.values[arg] for arg in op.args[1]]

        dst = args[0]
        srcs = args[1:]

        dst_descriptor  = dst._data
        src_descriptors = [src._data for src in srcs]

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
